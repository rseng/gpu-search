# https://github.com/upb-lea/ElectricGrid.jl

```console
docs/src/RL_Classical_Controllers_Merge.md:                          use_gpu = false)
docs/src/RL_Single_Agent.md:                          use_gpu = false);
docs/src/RL_Complex.md:                          use_gpu = false)
test/env_test_state.jl:    env = ElectricGridEnv(ts = ts, use_gpu = false, CM = [0 1;-1 0], num_sources = 1, num_loads = 1, verbosity = 0,parameters = parameters, maxsteps = length(t), action_delay = 1)
test/env_test_state.jl:    env = ElectricGridEnv(ts = 1e-6, use_gpu = false, CM = CM, num_sources = 2, num_loads = 1, parameters = parameters, maxsteps = 300, action_delay = 1, verbosity = 0)
test/runtests.jl:    #agent = CreateAgentDdpg(na = length(env.action_space), ns = length(env.state_space), use_gpu = false)
Project.toml:CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Project.toml:CUDA = "3.13.1"
examples/scripts/RL_Classical_Controllers_Merge.jl:    use_gpu = false)
examples/scripts/RL_Classical_Controllers_Merge.jl:                          use_gpu = false)
examples/scripts/RL_example.jl:                          use_gpu = false);
src/ElectricGrid.jl:using CUDA
src/electric_grid_env.jl:    gpu before
src/electric_grid_env.jl:- `use_gpu::Bool`: Flag if the simulation is done on gpu (if possible).
src/electric_grid_env.jl:    use_gpu=false,
src/electric_grid_env.jl:        if use_gpu
src/electric_grid_env.jl:        if use_gpu
src/electric_grid_env.jl:    if use_gpu
src/multi_controller.jl:                                        use_gpu = false)
src/agent_ddpg.jl:import CUDA: device
src/agent_ddpg.jl:function CreateAgentDdpg(;na, ns, batch_size = 32, use_gpu = true)
src/agent_ddpg.jl:                model = use_gpu ? CreateActor(na, ns) |> gpu : CreateActor(na, ns),
src/agent_ddpg.jl:                model = use_gpu ? CreateCritic(na, ns) |> gpu : CreateCritic(na, ns),
src/agent_ddpg.jl:                model = use_gpu ? CreateActor(na, ns) |> gpu : CreateActor(na, ns),
src/agent_ddpg.jl:                model = use_gpu ? CreateCritic(na, ns) |> gpu : CreateCritic(na, ns),

```