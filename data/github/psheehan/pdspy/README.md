# https://github.com/psheehan/pdspy

```console
bin/generate_surrogate_model.py:    # If a GPU is available, put the data there.
bin/generate_surrogate_model.py:    if torch.cuda.is_available():
bin/generate_surrogate_model.py:        inducing_points = inducing_points.cuda()
bin/generate_surrogate_model.py:        train_x = train_x.cuda()
bin/generate_surrogate_model.py:        train_y = train_y.cuda()
bin/generate_surrogate_model.py:        model = model.cuda()
bin/generate_surrogate_model.py:        likelihood = likelihood.cuda()
bin/generate_surrogate_model.py:    # Bring the data and models back from the GPU, if they were there.
bin/generate_surrogate_model.py:    if torch.cuda.is_available():

```
