# https://github.com/NREL/graph-env

```console
devtools/aws/example-tsp.yml:    #image: "public.ecr.aws/c9w8g6o2/rlmolecule:latest-gpu"
devtools/aws/example-tsp.yml:    #image: "rayproject/ray-ml:latest-gpu" # You can change this to latest-cpu if you don't need GPU support and want a faster startup
devtools/aws/example-tsp.yml:    image: "rayproject/ray-ml:1.12.0-py37-gpu" # You can change this to latest-cpu if you don't need GPU support and want a faster startup
devtools/aws/example-tsp.yml:    # image: rayproject/ray:latest-gpu   # use this one if you don't need ML dependencies, it's faster to pull
devtools/aws/example-tsp.yml:    # Example of running a GPU head with CPU workers
devtools/aws/example-tsp.yml:    # head_image: "rayproject/ray-ml:latest-gpu"
devtools/aws/example-tsp.yml:    # Allow Ray to automatically detect GPUs
devtools/aws/example-tsp.yml:        # The node type's CPU and GPU resources are auto-detected based on AWS instance type.
devtools/aws/example-tsp.yml:        # If desired, you can override the autodetected CPU and GPU resources advertised to the autoscaler.
devtools/aws/example-tsp.yml:        # For example, to mark a node type as having 1 CPU, 1 GPU, and 5 units of a resource called "custom", set
devtools/aws/example-tsp.yml:        # resources: {"CPU": 1, "GPU": 1, "custom": 5}
devtools/aws/example-tsp.yml:        # The node type's CPU and GPU resources are auto-detected based on AWS instance type.
devtools/aws/example-tsp.yml:        # If desired, you can override the autodetected CPU and GPU resources advertised to the autoscaler.
devtools/aws/example-tsp.yml:        # For example, to mark a node type as having 1 CPU, 1 GPU, and 5 units of a resource called "custom", set
devtools/aws/example-tsp.yml:        # resources: {"CPU": 1, "GPU": 1, "custom": 5}
devtools/aws/example-tsp.yml:    # that has the "nightly" (e.g. "rayproject/ray-ml:nightly-gpu") or uncomment the following line:
tests/conftest.py:          .resources(num_gpus=0)\
experiments/hallway/run_hallway.py:        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
experiments/hallway/run_hallway.py:        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
experiments/hallway/custom_env.py:        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
experiments/hallway/custom_env.py:        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
experiments/tsp/run_tsp_aws.py:parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs")
experiments/tsp/run_tsp_aws.py:        .resources(num_gpus=args.num_gpus)
experiments/tsp/run-tsp.sh:#SBATCH --gres=gpu:1
experiments/tsp/run-tsp.sh:    --num-gpus=1 \
experiments/tsp/run_tsp.py:parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs")
experiments/tsp/run_tsp.py:        .resources(num_gpus=args.num_gpus)

```
