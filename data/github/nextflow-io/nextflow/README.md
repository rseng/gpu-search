# https://github.com/nextflow-io/nextflow

```console
plugins/nf-azure/src/resources/nextflow/cloud/azure/az-locations.json:      "physicalLocation": "Nagpur",
plugins/nf-google/changelog.txt:- Fix support for serviceAccountEmail and GPU accelerator [7f7007a8] #7f7007a8
plugins/nf-google/changelog.txt:- Add support for GPU accelerator to Google Batch (#3056) [f34ad7f6] <Ben Sherman>
plugins/nf-google/src/test/nextflow/cloud/google/batch/GoogleBatchTaskHandlerTest.groovy:        def ACCELERATOR = new AcceleratorResource(request: 1, type: 'nvidia-tesla-v100')
plugins/nf-google/src/test/nextflow/cloud/google/batch/GoogleBatchTaskHandlerTest.groovy:            '/var/lib/nvidia/lib64:/usr/local/nvidia/lib64',
plugins/nf-google/src/test/nextflow/cloud/google/batch/GoogleBatchTaskHandlerTest.groovy:            '/var/lib/nvidia/bin:/usr/local/nvidia/bin'
plugins/nf-google/src/test/nextflow/cloud/google/batch/GoogleBatchTaskHandlerTest.groovy:        allocationPolicy.getInstances(0).getInstallGpuDrivers() == true
plugins/nf-google/src/test/nextflow/cloud/google/batch/GoogleBatchTaskHandlerTest.groovy:                getInstallGpuDrivers() >> true
plugins/nf-google/src/test/nextflow/cloud/google/batch/GoogleBatchTaskHandlerTest.groovy:        instancePolicyOrTemplate.getInstallGpuDrivers() == true
plugins/nf-google/src/test/nextflow/cloud/google/lifesciences/GoogleLifeSciencesHelperTest.groovy:        def acc = new AcceleratorResource(request: 4, type: 'nvidia-tesla-k80')
plugins/nf-google/src/test/nextflow/cloud/google/lifesciences/GoogleLifeSciencesHelperTest.groovy:            getVirtualMachine().getAccelerators()[0].getType()=='nvidia-tesla-k80'
plugins/nf-google/src/main/nextflow/cloud/google/batch/client/BatchConfig.groovy:    private boolean installGpuDrivers
plugins/nf-google/src/main/nextflow/cloud/google/batch/client/BatchConfig.groovy:    boolean getInstallGpuDrivers() { installGpuDrivers }
plugins/nf-google/src/main/nextflow/cloud/google/batch/client/BatchConfig.groovy:        result.installGpuDrivers = session.config.navigate('google.batch.installGpuDrivers',false)
plugins/nf-google/src/main/nextflow/cloud/google/batch/GoogleBatchTaskHandler.groovy:        // add nvidia specific driver paths
plugins/nf-google/src/main/nextflow/cloud/google/batch/GoogleBatchTaskHandler.groovy:        // see https://cloud.google.com/batch/docs/create-run-job#create-job-gpu
plugins/nf-google/src/main/nextflow/cloud/google/batch/GoogleBatchTaskHandler.groovy:        if( accel && accel.type.toLowerCase().startsWith('nvidia-') ) {
plugins/nf-google/src/main/nextflow/cloud/google/batch/GoogleBatchTaskHandler.groovy:                .addVolumes('/var/lib/nvidia/lib64:/usr/local/nvidia/lib64')
plugins/nf-google/src/main/nextflow/cloud/google/batch/GoogleBatchTaskHandler.groovy:                .addVolumes('/var/lib/nvidia/bin:/usr/local/nvidia/bin')
plugins/nf-google/src/main/nextflow/cloud/google/batch/GoogleBatchTaskHandler.groovy:        // https://cloud.google.com/batch/docs/create-run-job#create-job-gpu
plugins/nf-google/src/main/nextflow/cloud/google/batch/GoogleBatchTaskHandler.groovy:                .setInstallGpuDrivers( executor.config.getInstallGpuDrivers() )
plugins/nf-google/src/main/nextflow/cloud/google/batch/GoogleBatchTaskHandler.groovy:                instancePolicyOrTemplate.setInstallGpuDrivers(true)
plugins/nf-amazon/src/test/nextflow/cloud/aws/batch/AwsBatchTaskHandlerTest.groovy:        req.getContainerOverrides().getResourceRequirements().find { it.type=='GPU'}.getValue() == '2'
plugins/nf-amazon/src/main/nextflow/cloud/aws/batch/AwsBatchTaskHandler.groovy:            resources << new ResourceRequirement().withType(ResourceType.GPU).withValue(accelerator.request.toString())
plugins/nf-amazon/src/main/nextflow/cloud/aws/AwsClientFactory.groovy:import com.amazonaws.services.logs.AWSLogsAsyncClientBuilder
plugins/nf-amazon/src/main/nextflow/cloud/aws/AwsClientFactory.groovy:        final builder = AWSLogsAsyncClientBuilder
changelog.txt:- Update google batch java sdk, add serviceAccountEmail and installGpuDrivers (#3324) [7f7007a8]
changelog.txt:- Add support for GPU accelerator to Google Batch (#3056) [f34ad7f6] <Ben Sherman>
changelog.txt:- Fix for feature 1647 that allows non gpu k8s accelerator resources. (#1649) [c1efa29d]
changelog.txt:- Remove deprecated gpu directive [f48b2c24]
changelog.txt:- Add Experimental support for gpu resources #997
docs/reference/process.md:The `accelerator` directive allows you to request hardware accelerators (e.g. GPUs) for the task execution. For example:
docs/reference/process.md:    accelerator 4, type: 'nvidia-tesla-k80'
docs/reference/process.md:    your_gpu_enabled --command --line
docs/reference/process.md:The above examples will request 4 GPUs of type `nvidia-tesla-k80`.
docs/reference/process.md:- [Google Cloud](https://cloud.google.com/compute/docs/gpus/)
docs/reference/process.md:- [Kubernetes](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/#clusters-containing-different-types-of-gpus)
docs/reference/process.md:The accelerator `type` option is not supported for AWS Batch. You can control the accelerator type indirectly through the allowed instance types in your Compute Environment. See the [AWS Batch FAQs](https://aws.amazon.com/batch/faqs/?#GPU_Scheduling_) for more information.
docs/google.md:To use an instance template with GPUs, you must also set the `google.batch.installGpuDrivers` config option to `true`.
docs/google.md:- Compute resources in Google Cloud are subject to [resource quotas](https://cloud.google.com/compute/quotas), which may affect your ability to run pipelines at scale. You can request quota increases, and your quotas may automatically increase over time as you use the platform. In particular, GPU quotas are initially set to 0, so you must explicitly request a quota increase in order to use GPUs. You can initially request an increase to 1 GPU at a time, and after one billing cycle you may be able to increase it further.
modules/nextflow/src/test/groovy/nextflow/container/ApptainerBuilderTest.groovy:                .addEnv('CUDA_VISIBLE_DEVICES')
modules/nextflow/src/test/groovy/nextflow/container/ApptainerBuilderTest.groovy:                .runCommand == 'set +u; env - PATH="$PATH" ${TMP:+APPTAINERENV_TMP="$TMP"} ${TMPDIR:+APPTAINERENV_TMPDIR="$TMPDIR"} ${CUDA_VISIBLE_DEVICES:+APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"} apptainer exec --no-home --pid -B "$NXF_TASK_WORKDIR" busybox'
modules/nextflow/src/test/groovy/nextflow/container/SingularityBuilderTest.groovy:                .addEnv('CUDA_VISIBLE_DEVICES')
modules/nextflow/src/test/groovy/nextflow/container/SingularityBuilderTest.groovy:                .runCommand == 'set +u; env - PATH="$PATH" ${TMP:+SINGULARITYENV_TMP="$TMP"} ${TMPDIR:+SINGULARITYENV_TMPDIR="$TMPDIR"} ${CUDA_VISIBLE_DEVICES:+SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"} singularity exec --no-home --pid -B "$NXF_TASK_WORKDIR" busybox'
modules/nextflow/src/test/groovy/nextflow/executor/HyperQueueExecutorTest.groovy:            #HQ --resource gpus=2
modules/nextflow/src/test/groovy/nextflow/executor/res/AcceleratorResourceTest.groovy:    def 'should create a gpu resource' () {
modules/nextflow/src/test/groovy/nextflow/executor/res/AcceleratorResourceTest.groovy:        [limit: 3, type: 'nvidia']  | 3     | 3     | 'nvidia' | null
modules/nextflow/src/test/groovy/nextflow/extension/BufferOpTest.groovy:    def testBufferOpenClose() {
modules/nextflow/src/test/groovy/nextflow/processor/TaskConfigTest.groovy:    def 'should get gpu resources' () {
modules/nextflow/src/test/groovy/nextflow/processor/TaskConfigTest.groovy:        process.accelerator 5, limit: 10, type: 'nvidia'
modules/nextflow/src/test/groovy/nextflow/processor/TaskConfigTest.groovy:        res.type == 'nvidia'
modules/nextflow/src/test/groovy/nextflow/script/ProcessConfigTest.groovy:                'withLabel:gpu.+'  : [ cpus: 4 ],
modules/nextflow/src/test/groovy/nextflow/script/ProcessConfigTest.groovy:        process.applyConfigSelectorWithLabels(settings, ['gpu-1'])
modules/nextflow/src/test/groovy/nextflow/k8s/model/PodNodeSelectorTest.groovy:        'gpu,intel'         | [gpu:'true',intel: 'true']
modules/nextflow/src/test/groovy/nextflow/k8s/model/PodSpecBuilderTest.groovy:                requests: ['foo.org/gpu':5, cpu:8, memory:'100Gi', 'ephemeral-storage':'10Gi'],
modules/nextflow/src/test/groovy/nextflow/k8s/model/PodSpecBuilderTest.groovy:                limits: ['foo.org/gpu':10, memory:'100Gi', 'ephemeral-storage':'10Gi']
modules/nextflow/src/test/groovy/nextflow/k8s/model/PodSpecBuilderTest.groovy:        _ * opts.getNodeSelector() >> new PodNodeSelector(gpu:true, queue: 'fast')
modules/nextflow/src/test/groovy/nextflow/k8s/model/PodSpecBuilderTest.groovy:        pod.spec.nodeSelector == [gpu: 'true', queue: 'fast']
modules/nextflow/src/test/groovy/nextflow/k8s/model/PodSpecBuilderTest.groovy:        res.requests == ['nvidia.com/gpu': 2]
modules/nextflow/src/test/groovy/nextflow/k8s/model/PodSpecBuilderTest.groovy:        res.limits == ['nvidia.com/gpu': 5]
modules/nextflow/src/test/groovy/nextflow/k8s/model/PodSpecBuilderTest.groovy:        res.requests == ['foo.com/gpu': 5]
modules/nextflow/src/test/groovy/nextflow/k8s/model/PodSpecBuilderTest.groovy:        res.limits == ['foo.com/gpu': 5]
modules/nextflow/src/test/groovy/nextflow/k8s/model/PodSpecBuilderTest.groovy:        res.requests == ['foo.org/gpu': 5]
modules/nextflow/src/test/groovy/nextflow/k8s/model/PodSpecBuilderTest.groovy:        res.requests == [cpu: 2, 'foo.org/gpu': 5]
modules/nextflow/src/test/groovy/nextflow/k8s/model/PodSpecBuilderTest.groovy:        res.requests == [cpu: 2, 'foo.org/gpu': 5]
modules/nextflow/src/test/groovy/nextflow/k8s/model/PodSpecBuilderTest.groovy:        res.limits == ['foo.org/gpu': 10]
modules/nextflow/src/main/groovy/nextflow/executor/HyperQueueExecutor.groovy:            result << '--resource' << "gpus=${task.config.getAccelerator().limit}".toString()
modules/nextflow/src/main/groovy/nextflow/k8s/model/PodSpecBuilder.groovy:        def type = accelerator.type ?: 'nvidia.com'
modules/nextflow/src/main/groovy/nextflow/k8s/model/PodSpecBuilder.groovy:        // Assume we're using GPU and update as necessary.
modules/nextflow/src/main/groovy/nextflow/k8s/model/PodSpecBuilder.groovy:        type += '/gpu'

```
