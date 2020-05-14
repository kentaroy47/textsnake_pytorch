import torch
import torch.nn as nn
import torch.nn.functional as F

def focal_loss(pred, gt, alpha=3):
  ''' 
    alpha: FocalLoss論文のアルファパラメータ。
    alpha = 1の時はCEと同様のロス。
    alpha = 2で論文の値。データ中に物体が少ないため、3がよさそう。
  '''
  pred = pred.unsqueeze(1).float()
  gt = gt.unsqueeze(1).float()

  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()
  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred + 1e-12) * torch.pow(1 - pred, alpha) * pos_inds
  neg_loss = torch.log(1 - pred + 1e-12) * torch.pow(pred, alpha) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

class TextLoss(nn.Module):

    def __init__(self, focal=False):
        super().__init__()
        self.focal = focal
    
    # Online hard negative miningの実装
    # Positive vs negativeが1:3になるよう、最も確信度の低い上位negative結果のみロス計算に回す。
    def ohem(self, predict, target, train_mask, negative_ratio=3.):
        # 出力をposとnegに分離
        pos = (target * train_mask).byte()
        neg = ((1 - target) * train_mask).byte()
        # postiveの個数をposのsumを取ることで計算
        n_pos = pos.float().sum()
        
        #n_posに応じてn_negを決定。CEでロス計算。
        if n_pos.item() > 0:
            loss_pos = F.cross_entropy(predict[pos], target[pos], reduction='sum')
            loss_neg = F.cross_entropy(predict[neg], target[neg], reduction='none')
            n_neg = min(int(neg.float().sum().item()), int(negative_ratio * n_pos.float()))
        else:
            loss_pos = 0.
            loss_neg = F.cross_entropy(predict[neg], target[neg], reduction='none')
            n_neg = 100
        # Top-kでloss_negの上位ロス以外フィルタリング
        loss_neg, _ = torch.topk(loss_neg, n_neg)
        # Ltrを計算。
        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    def forward(self, input, tr_mask, tcl_mask, sin_map, cos_map, radii_map, train_mask):
        """
        calculate textsnake loss
        :param input: (Variable), network predict, (BS, 7, H, W)
        :param tr_mask: (Variable), TR target, (BS, H, W)
        :param tcl_mask: (Variable), TCL target, (BS, H, W)
        :param sin_map: (Variable), sin target, (BS, H, W)
        :param cos_map: (Variable), cos target, (BS, H, W)
        :param radii_map: (Variable), radius target, (BS, H, W)
        :param train_mask: (Variable), training mask, (BS, H, W)
        :return: loss_tr, loss_tcl, loss_radii, loss_sin, loss_cos
        """
        # 出力マップからそれぞれのpredictionを分離
        # マスク処理を高速、簡単にするために一次元にそれぞれ展開
        # contiguous()でメモリ上で要素順に並べる
        # https://qiita.com/kenta1984/items/d68b72214ce92beebbe2
        tr_pred = input[:, :2].permute(0, 2, 3, 1).contiguous().view(-1, 2)  # (BSxHxW, 2)
        tcl_pred = input[:, 2:4].permute(0, 2, 3, 1).contiguous().view(-1, 2)  # (BSxHxW, 2)
        sin_pred = input[:, 4].contiguous().view(-1)  # (BSxHxW,)
        cos_pred = input[:, 5].contiguous().view(-1)  # (BSxHxW,)
        radii_pred = input[:, 6].contiguous().view(-1)  # (BSxHxW,)
        
        # regularize sin and cos: sum to 1
        scale = torch.sqrt(1.0 / (sin_pred ** 2 + cos_pred ** 2))
        sin_pred = sin_pred * scale
        cos_pred = cos_pred * scale
        
        # ラベルも同様に一次元に展開する
        train_mask = train_mask.view(-1)  # (BSxHxW,)
        tr_mask = tr_mask.contiguous().view(-1)
        tcl_mask = tcl_mask.contiguous().view(-1)
        radii_map = radii_map.contiguous().view(-1)
        sin_map = sin_map.contiguous().view(-1)
        cos_map = cos_map.contiguous().view(-1)

        # Online hard miningによってnegative-positiveのバランスを取り、trロスを導出
        if not focal:
            loss_tr = self.ohem(tr_pred, tr_mask.long(), train_mask.long())
        else:
            # Focal loss
            tr_train_mask = train_mask * tr_mask
            loss_tr = focal_loss(tr_pred, tr_train_mask.long())

        loss_tcl = 0.
        # TCLや他Lregはtr_mask*train_maskでマスクしロスを計算する。
        tr_train_mask = train_mask * tr_mask
       
        if tr_train_mask.sum().item() > 0:
            if not focal:
                # もし物体が含まれるならmaskした部分のみでロス計算。普通のCE。
                loss_tcl = F.cross_entropy(tcl_pred[tr_train_mask], tcl_mask[tr_train_mask].long())
            else:
                loss_tcl = focal_loss(tcl_pred[tr_train_mask], tcl_mask[tr_train_mask].long())

        # geometry losses
        # マスクした部分でSmooth L1 lossを計算。
        loss_radii, loss_sin, loss_cos = 0., 0., 0.
        tcl_train_mask = train_mask * tcl_mask
        if tcl_train_mask.sum().item() > 0:
            ones = radii_map.new(radii_pred[tcl_mask].size()).fill_(1.).float()
            loss_radii = F.smooth_l1_loss(radii_pred[tcl_mask] / radii_map[tcl_mask], ones)
            loss_sin = F.smooth_l1_loss(sin_pred[tcl_mask], sin_map[tcl_mask])
            loss_cos = F.smooth_l1_loss(cos_pred[tcl_mask], cos_map[tcl_mask])

        return loss_tr, loss_tcl, loss_radii, loss_sin, loss_cos
