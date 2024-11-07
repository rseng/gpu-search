# https://github.com/ABard0/MDGF-MCEC

```console
code/circRNA2Disease/MDGF-MCEC.py:###THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python
code/circRNA2Disease/MDGF-MCEC.py:    # one_index = train_data[2][0FS].cuda().t().tolist()
code/circRNA2Disease/MDGF-MCEC.py:    # zero_index = train_data[2][1].cuda().t().tolist()
code/circRNA2Disease/MDGF-MCEC.py:        loss = loss(score, train_data['md_p'].cuda())
code/circRNA2Disease/MDGF-MCEC.py:    #         model.cuda()
code/circRNA2Disease/MDGF-MCEC.py:    #     model.cuda()
code/circRNA2Disease/model.py:        x_m_f1 = torch.relu(self.gcn_x1_f(x_m.cuda(), data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][
code/circRNA2Disease/model.py:            data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))
code/circRNA2Disease/model.py:        x_m_f2 = torch.relu(self.gcn_x2_f(x_m_f1, data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][
code/circRNA2Disease/model.py:            data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))
code/circRNA2Disease/model.py:        x_m_g1 = torch.relu(self.gcn_x1_s(x_m.cuda(), data['mm_g']['edges'].cuda(), data['mm_g']['data_matrix'][
code/circRNA2Disease/model.py:            data['mm_g']['edges'][0], data['mm_g']['edges'][1]].cuda()))
code/circRNA2Disease/model.py:        x_m_g2 = torch.relu(self.gcn_x2_s(x_m_g1, data['mm_g']['edges'].cuda(), data['mm_g']['data_matrix'][
code/circRNA2Disease/model.py:            data['mm_g']['edges'][0], data['mm_g']['edges'][1]].cuda()))
code/circRNA2Disease/model.py:        y_d_f1 = torch.relu(self.gcn_y1_f(x_d.cuda(), data['dd_f']['edges'].cuda(), data['dd_f']['data_matrix'][
code/circRNA2Disease/model.py:            data['dd_f']['edges'][0], data['dd_f']['edges'][1]].cuda()))
code/circRNA2Disease/model.py:        y_d_f2 = torch.relu(self.gcn_y2_f(y_d_f1, data['dd_f']['edges'].cuda(), data['dd_f']['data_matrix'][
code/circRNA2Disease/model.py:            data['dd_f']['edges'][0], data['dd_f']['edges'][1]].cuda()))
code/circRNA2Disease/model.py:        y_d_g1 = torch.relu(self.gcn_y1_s(x_d.cuda(), data['dd_g']['edges'].cuda(), data['dd_g']['data_matrix'][
code/circRNA2Disease/model.py:            data['dd_g']['edges'][0], data['dd_g']['edges'][1]].cuda()))
code/circRNA2Disease/model.py:        y_d_g2 = torch.relu(self.gcn_y2_s(y_d_g1, data['dd_g']['edges'].cuda(), data['dd_g']['data_matrix'][
code/circRNA2Disease/model.py:            data['dd_g']['edges'][0], data['dd_g']['edges'][1]].cuda()))

```
