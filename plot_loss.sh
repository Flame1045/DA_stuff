log=$1

python3 tools/analysis_tools/analyze_logs.py plot_curve outputs/$1  --keys da_loss0 enc_loss_cls enc_loss_bbox enc_loss_iou loss_cls \
        loss_bbox loss_iou d4.loss_bbox_aux0 d4.loss_iou_aux0 \
        d0.loss_cls d0.loss_bbox d0.loss_iou d1.loss_cls d1.loss_bbox d1.loss_iou d2.loss_cls d2.loss_bbox \
        d2.loss_iou d3.loss_cls d3.loss_bbox d3.loss_iou d4.loss_cls d4.loss_bbox d4.loss_iou loss_rpn_cls \
        loss_rpn_bbox loss_cls0 loss_bbox0 loss_cls_aux0 loss_bbox_aux0 loss_iou_aux0 d0.loss_cls_aux0 \
        d0.loss_bbox_aux0 d0.loss_iou_aux0 d1.loss_cls_aux0 d1.loss_bbox_aux0 d1.loss_iou_aux0 d2.loss_cls_aux0 \
        d2.loss_bbox_aux0 d2.loss_iou_aux0 d3.loss_cls_aux0 d3.loss_bbox_aux0 d3.loss_iou_aux0 d4.loss_cls_aux0 \
        


