# https://github.com/xenon-middleware/xenon

```console
src/test/java/nl/esciencecenter/xenon/adaptors/schedulers/gridengine/GridEngineXmlParserTest.java:        assertArrayEquals(new Object[] { "all.q", "das3.q", "disabled.q", "fat.q", "gpu.q" }, queues);
src/test/java/nl/esciencecenter/xenon/adaptors/schedulers/gridengine/GridEngineUtilsTest.java:        description.setSchedulerArguments("-l gpu=1");
src/test/java/nl/esciencecenter/xenon/adaptors/schedulers/gridengine/GridEngineUtilsTest.java:        String expected = "#!/bin/sh\n" + "#$ -S /bin/sh\n" + "#$ -N xenon\n" + "#$ -l h_rt=00:15:00\n" + "#$ -l gpu=1\n" + "#$ -o /dev/null\n"
src/test/java/nl/esciencecenter/xenon/adaptors/schedulers/ScriptingParserTest.java:                + "Features=gpunode DelayBoot=00:00:00\n" + "Gres=(null) Reservation=(null)\n"
src/test/java/nl/esciencecenter/xenon/adaptors/schedulers/ScriptingParserTest.java:                + "GresTypes               = (null)\n" + "GpuFreqDef              = high,memory=high\n" + "GroupUpdateForce        = 1\n"
src/test/java/nl/esciencecenter/xenon/adaptors/schedulers/ScriptingParserTest.java:                + "GetEnvTimeout           = 2 sec\n" + "GresTypes               = (null)\n" + "GpuFreqDef              = high,memory=high\n"
src/test/java/nl/esciencecenter/xenon/adaptors/schedulers/slurm/SlurmUtilsTest.java:        description.setSchedulerArguments("--gres=gpu:1", "-C TitanX");
src/test/java/nl/esciencecenter/xenon/adaptors/schedulers/slurm/SlurmUtilsTest.java:                + "#SBATCH --output=/dev/null\n" + "#SBATCH --error=/dev/null\n" + "#SBATCH --gres=gpu:1\n" + "#SBATCH -C TitanX\n" + "\n"
src/test/resources/fixtures/gridengine/jobs.xml:      <queue_name>gpu.q@node025.cm.cluster</queue_name>
src/test/resources/fixtures/gridengine/detailed-job.xml:          <VA_value>/home/ceriel/bin:/home/ceriel/jdk1.7.0_04/bin:/usr/local/package/apache-ant-1.6.5/bin:/cm/shared/apps/gcc/4.4.6/bin:/usr/lib64/qt-3.3/bin:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/sbin:/usr/sbin:/cm/shared/apps/sge/6.2u5p2/bin/lx26-amd64:/cm/shared/package/reserve.sge/bin:/cm/shared/apps/cuda40/toolkit/4.0.17/bin:/cm/shared/apps/cuda40/sdk/4.0.17/C/bin/linux/release:/cm/local/apps/cuda50/libs/304.54/usr/bin:/cm/shared/apps/slurm/2.2.7/bin:/cm/shared/apps/slurm/2.2.7/sbin</VA_value>
src/test/resources/fixtures/gridengine/detailed-job.xml:          <MES_message>queue instance &quot;gpu.q@node025.cm.cluster&quot; dropped because it is full</MES_message>
src/test/resources/fixtures/gridengine/queues.xml:    <name>gpu.q</name>
src/test/resources/fixtures/gridengine/jobs-wrong-schema.xml:      <queue_name>gpu.q@node025.cm.cluster</queue_name>
src/test/resources/fixtures/gridengine/jobs-no-schema.xml:      <queue_name>gpu.q@node025.cm.cluster</queue_name>

```
