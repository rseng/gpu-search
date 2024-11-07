# https://github.com/pixixiaonaogou/MLSDR

```console
second_stage_fusion_utils.py:         ] = net(((clinic_img_tensor).cuda(), dermoscopy_img_tensor.cuda()))
second_stage_fusion_utils.py:            ((clinic_img_tensor).cuda(), dermoscopy_img_tensor.cuda()))
main_cmv2.py:os.environ['CUDA_VISIBLE_DEVICES'] = '0'
main_cmv2.py:        clinic_image = clinic_image.cuda()
main_cmv2.py:        derm_image   = derm_image.cuda()
main_cmv2.py:        meta_data    = meta_data.cuda()
main_cmv2.py:        diagnosis_label = label[0].long().cuda()
main_cmv2.py:        pn_label = label[1].long().cuda()
main_cmv2.py:        str_label = label[2].long().cuda()
main_cmv2.py:        pig_label = label[3].long().cuda()
main_cmv2.py:        rs_label = label[4].long().cuda()
main_cmv2.py:        dag_label = label[5].long().cuda()
main_cmv2.py:        bwv_label = label[6].long().cuda()
main_cmv2.py:        vs_label = label[7].long().cuda()
main_cmv2.py:        clinic_image = clinic_image.cuda()
main_cmv2.py:        derm_image   = derm_image.cuda()
main_cmv2.py:        meta_data    = meta_data.cuda()
main_cmv2.py:        diagnosis_label = label[0].long().cuda()
main_cmv2.py:        pn_label = label[1].long().cuda()
main_cmv2.py:        str_label = label[2].long().cuda()
main_cmv2.py:        pig_label = label[3].long().cuda()
main_cmv2.py:        rs_label = label[4].long().cuda()
main_cmv2.py:        dag_label = label[5].long().cuda()
main_cmv2.py:        bwv_label = label[6].long().cuda()
main_cmv2.py:        vs_label = label[7].long().cuda()
main_cmv2.py:        net = FusionNet(class_list).cuda()
model.py:#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
utils.py:      torch.cuda.manual_seed(seed)
utils.py:      torch.cuda.manual_seed_all(seed)

```
