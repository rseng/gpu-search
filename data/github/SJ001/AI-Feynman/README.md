# https://github.com/SJ001/AI-Feynman

```console
README.md:This example will get solved in about 10-30 minutes depending on what computer you have and whether you have a GPU.
aifeynman/S_NN_get_gradients.py:is_cuda = torch.cuda.is_available()
aifeynman/S_NN_get_gradients.py:        is_cuda = torch.cuda.is_available()
aifeynman/S_NN_get_gradients.py:        if is_cuda:
aifeynman/S_NN_get_gradients.py:            pts = pts.float().cuda()
aifeynman/S_NN_get_gradients.py:            model = model.cuda()
aifeynman/S_NN_get_gradients.py:            grad_weights = grad_weights.cuda()
aifeynman/S_compositionality.py:is_cuda = torch.cuda.is_available()
aifeynman/S_compositionality.py:                if is_cuda:
aifeynman/S_compositionality.py:                    dt = torch.tensor(dt).float().cuda().view(1,len(dt))
aifeynman/S_compositionality.py:                    dt = torch.cat((torch.tensor([np.zeros(len(dt[0]))]).float().cuda(),dt), 0)
aifeynman/S_compositionality.py:                    error = torch.tensor(data[:,-1][i]).cuda()-model(dt)[1:]
aifeynman/S_remove_input_neuron.py:is_cuda = torch.cuda.is_available()
aifeynman/S_remove_input_neuron.py:    if is_cuda:
aifeynman/S_remove_input_neuron.py:        net.linear1.bias = nn.Parameter(net.linear1.bias+torch.tensor(ct_median*removed_weights).float().cuda())
aifeynman/S_gen_sym.py:is_cuda = torch.cuda.is_available()
aifeynman/S_gen_sym.py:                if is_cuda:
aifeynman/S_gen_sym.py:                    dt = torch.tensor(dt_).float().cuda().view(1,len(dt_))
aifeynman/S_gen_sym.py:                    dt = torch.cat((torch.tensor([np.zeros(len(dt[0]))]).float().cuda(),dt), 0)
aifeynman/S_gen_sym.py:                    error = torch.tensor(data[:,-1][i]).cuda()-model(dt[:,:-1])[1:]
aifeynman/S_gradient_decomposition.py:        device = 'cuda' if model.is_cuda else 'cpu'
aifeynman/S_gradient_decomposition.py:    is_cuda = torch.cuda.is_available()
aifeynman/S_gradient_decomposition.py:    if is_cuda:
aifeynman/S_gradient_decomposition.py:        pts = pts.cuda()
aifeynman/S_gradient_decomposition.py:        model = model.cuda()
aifeynman/S_gradient_decomposition.py:        grad_weights = grad_weights.cuda()
aifeynman/S_gradient_decomposition.py:    is_cuda = X.is_cuda
aifeynman/S_gradient_decomposition.py:    device = 'cuda' if is_cuda else 'cpu'
aifeynman/S_gradient_decomposition.py:    is_cuda = torch.cuda.is_available()
aifeynman/S_gradient_decomposition.py:    if is_cuda:
aifeynman/S_gradient_decomposition.py:        X = X.cuda()
aifeynman/S_gradient_decomposition.py:        y = y.cuda()
aifeynman/S_gradient_decomposition.py:        model = model.cuda()
aifeynman/S_gradient_decomposition.py:        # print('Using cuda')
aifeynman/S_gradient_decomposition.py:    is_cuda = torch.cuda.is_available()
aifeynman/S_gradient_decomposition.py:    if is_cuda:
aifeynman/S_gradient_decomposition.py:        X = X.cuda()
aifeynman/S_gradient_decomposition.py:        y = y.cuda()
aifeynman/S_gradient_decomposition.py:        model = model.cuda()
aifeynman/S_NN_train.py:is_cuda = torch.cuda.is_available()
aifeynman/S_NN_train.py:        if is_cuda:
aifeynman/S_NN_train.py:            factors = factors.cuda()
aifeynman/S_NN_train.py:        if is_cuda:
aifeynman/S_NN_train.py:            product = product.cuda()
aifeynman/S_NN_train.py:        if is_cuda:
aifeynman/S_NN_train.py:            model_feynman = SimpleNet(n_variables).cuda()
aifeynman/S_NN_train.py:                    if is_cuda:
aifeynman/S_NN_train.py:                        fct = data[0].float().cuda()
aifeynman/S_NN_train.py:                        prd = data[1].float().cuda()
aifeynman/S_separability.py:is_cuda = torch.cuda.is_available()
aifeynman/S_separability.py:        if is_cuda:
aifeynman/S_separability.py:            factors = factors.cuda()
aifeynman/S_separability.py:        if is_cuda:
aifeynman/S_separability.py:            product = product.cuda()
aifeynman/S_separability.py:        if is_cuda:
aifeynman/S_separability.py:            model = SimpleNet(n_variables).cuda()
aifeynman/S_separability.py:        if is_cuda:
aifeynman/S_separability.py:            factors = factors.cuda()
aifeynman/S_separability.py:        if is_cuda:
aifeynman/S_separability.py:            product = product.cuda()
aifeynman/S_separability.py:        if is_cuda:
aifeynman/S_separability.py:            model = SimpleNet(n_variables).cuda()
aifeynman/S_separability.py:        if is_cuda:
aifeynman/S_separability.py:            factors = factors.cuda()
aifeynman/S_separability.py:        if is_cuda:
aifeynman/S_separability.py:            product = product.cuda()
aifeynman/S_separability.py:        if is_cuda:
aifeynman/S_separability.py:            model = SimpleNet(n_variables).cuda()
aifeynman/S_separability.py:        if is_cuda:
aifeynman/S_separability.py:            factors = factors.cuda()
aifeynman/S_separability.py:        if is_cuda:
aifeynman/S_separability.py:            product = product.cuda()
aifeynman/S_separability.py:        if is_cuda:
aifeynman/S_separability.py:            model = SimpleNet(n_variables).cuda()
aifeynman/S_NN_eval.py:is_cuda = torch.cuda.is_available()
aifeynman/S_NN_eval.py:        if is_cuda:
aifeynman/S_NN_eval.py:            factors = factors.cuda()
aifeynman/S_NN_eval.py:        if is_cuda:
aifeynman/S_NN_eval.py:            product = product.cuda()
aifeynman/S_NN_eval.py:        if is_cuda:
aifeynman/S_NN_eval.py:            factors_val = factors_val.cuda()
aifeynman/S_NN_eval.py:        if is_cuda:
aifeynman/S_NN_eval.py:            product_val = product_val.cuda()
aifeynman/S_NN_eval.py:        if is_cuda:
aifeynman/S_NN_eval.py:            model = SimpleNet(n_variables).cuda()
aifeynman/S_symmetry.py:is_cuda = torch.cuda.is_available()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            factors = factors.cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            product = product.cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            model = SimpleNet(n_variables).cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            factors = factors.cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            product = product.cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            model = SimpleNet(n_variables).cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            factors = factors.cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            product = product.cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            model = SimpleNet(n_variables).cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            factors = factors.cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            product = product.cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            model = SimpleNet(n_variables).cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            factors = factors.cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            product = product.cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            model = SimpleNet(n_variables).cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            factors = factors.cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            product = product.cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            model = SimpleNet(n_variables).cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            factors = factors.cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:                product = product.cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            model = SimpleNet(n_variables).cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            factors = factors.cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            product = product.cuda()
aifeynman/S_symmetry.py:        if is_cuda:
aifeynman/S_symmetry.py:            model = SimpleNet(n_variables).cuda()

```
