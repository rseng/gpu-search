# https://github.com/lsds/Neptune

```console
docs/sql-programming-guide.md:* `configure` (`GenericUDF`, `GenericUDTF`, and `GenericUDAFEvaluator`) is a function to initialize
docs/sql-programming-guide.md:* `close` (`GenericUDF` and `GenericUDAFEvaluator`) is a function to release associated resources.
docs/sql-programming-guide.md:* `reset` (`GenericUDAFEvaluator`) is a function to re-initialize aggregation for reusing the same aggregation.
docs/sql-programming-guide.md:* `getWindowingEvaluator` (`GenericUDAFEvaluator`) is a function to optimize aggregation by evaluating
docs/running-on-mesos.md:  <td><code>spark.mesos.gpus.max</code></td>
docs/running-on-mesos.md:    Set the maximum number GPU resources to acquire for this job. Note that executors will still launch when no GPU resources are found
NOTICE:The inverse error function implementation in the Erf class is based on CUDA
NOTICE:and published in GPU Computing Gems, volume 2, 2010.
launcher/src/test/java/org/apache/spark/launcher/ChildProcAppHandleSuite.java:  public void testProcMonitorWithOutputRedirection() throws Exception {
launcher/src/test/java/org/apache/spark/launcher/ChildProcAppHandleSuite.java:  public void testProcMonitorWithLogRedirection() throws Exception {
resource-managers/mesos/src/test/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackendSuite.scala:  test("mesos does not acquire gpus if not specified") {
resource-managers/mesos/src/test/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackendSuite.scala:    val gpus = backend.getResource(taskInfos.head.getResourcesList, "gpus")
resource-managers/mesos/src/test/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackendSuite.scala:    assert(gpus == 0.0)
resource-managers/mesos/src/test/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackendSuite.scala:  test("mesos does not acquire more than spark.mesos.gpus.max") {
resource-managers/mesos/src/test/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackendSuite.scala:    val maxGpus = 5
resource-managers/mesos/src/test/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackendSuite.scala:    setBackend(Map("spark.mesos.gpus.max" -> maxGpus.toString))
resource-managers/mesos/src/test/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackendSuite.scala:    offerResources(List(Resources(executorMemory, 1, maxGpus + 1)))
resource-managers/mesos/src/test/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackendSuite.scala:    val gpus = backend.getResource(taskInfos.head.getResourcesList, "gpus")
resource-managers/mesos/src/test/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackendSuite.scala:    assert(gpus == maxGpus)
resource-managers/mesos/src/test/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackendSuite.scala:  private case class Resources(mem: Int, cpus: Int, gpus: Int = 0)
resource-managers/mesos/src/test/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackendSuite.scala:      createOffer(s"o${i + startId}", s"s${i + startId}", offer.mem, offer.cpus, None, offer.gpus)}
resource-managers/mesos/src/test/scala/org/apache/spark/scheduler/cluster/mesos/Utils.scala:                   gpus: Int = 0,
resource-managers/mesos/src/test/scala/org/apache/spark/scheduler/cluster/mesos/Utils.scala:    if (gpus > 0) {
resource-managers/mesos/src/test/scala/org/apache/spark/scheduler/cluster/mesos/Utils.scala:        .setName("gpus")
resource-managers/mesos/src/test/scala/org/apache/spark/scheduler/cluster/mesos/Utils.scala:        .setScalar(Scalar.newBuilder().setValue(gpus))
resource-managers/mesos/src/test/scala/org/apache/spark/scheduler/cluster/mesos/MesosSchedulerUtilsSuite.scala:    val offerAttribs = Map("gpus" -> Value.Scalar.newBuilder().setValue(3).build())
resource-managers/mesos/src/test/scala/org/apache/spark/scheduler/cluster/mesos/MesosSchedulerUtilsSuite.scala:    val ltConstraint = utils.parseConstraintString("gpus:2")
resource-managers/mesos/src/test/scala/org/apache/spark/scheduler/cluster/mesos/MesosSchedulerUtilsSuite.scala:    val eqConstraint = utils.parseConstraintString("gpus:3")
resource-managers/mesos/src/test/scala/org/apache/spark/scheduler/cluster/mesos/MesosSchedulerUtilsSuite.scala:    val gtConstraint = utils.parseConstraintString("gpus:4")
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosSchedulerUtils.scala:    val maxGpus = conf.getInt("spark.mesos.gpus.max", 0)
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosSchedulerUtils.scala:    if (maxGpus > 0) {
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosSchedulerUtils.scala:      fwInfoBuilder.addCapabilities(Capability.newBuilder().setType(Capability.Type.GPU_RESOURCES))
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackend.scala:  private val maxGpus = conf.getInt("spark.mesos.gpus.max", 0)
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackend.scala:  private val gpusByTaskId = new mutable.HashMap[String, Int]
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackend.scala:  private var totalGpusAcquired = 0
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackend.scala:          val taskGPUs = Math.min(
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackend.scala:            Math.max(0, maxGpus - totalGpusAcquired), getResource(resources, "gpus").toInt)
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackend.scala:            partitionTaskResources(resources, taskCPUs, taskMemory, taskGPUs)
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackend.scala:          if (taskGPUs > 0) {
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackend.scala:            totalGpusAcquired += taskGPUs
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackend.scala:            gpusByTaskId(taskId) = taskGPUs
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackend.scala:      taskGPUs: Int)
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackend.scala:    val (afterGPUResources, gpuResourcesToUse) =
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackend.scala:      partitionResources(afterMemResources.asJava, "gpus", taskGPUs)
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackend.scala:      partitionPortResources(nonZeroPortValuesFromConfig(sc.conf), afterGPUResources)
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackend.scala:      cpuResourcesToUse ++ memResourcesToUse ++ portResourcesToUse ++ gpuResourcesToUse)
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackend.scala:        // Also remove the gpus we have remembered for this task, if it's in the hashmap
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackend.scala:        for (gpus <- gpusByTaskId.get(taskId)) {
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackend.scala:          totalGpusAcquired -= gpus
resource-managers/mesos/src/main/scala/org/apache/spark/scheduler/cluster/mesos/MesosCoarseGrainedSchedulerBackend.scala:          gpusByTaskId -= taskId
sql/hive/src/test/resources/ql/src/test/queries/clientpositive/windowing_udaf2.q:create temporary function mysum as 'org.apache.hadoop.hive.ql.udf.generic.GenericUDAFSum';
sql/hive/src/test/resources/ql/src/test/queries/clientpositive/create_genericudaf.q:CREATE TEMPORARY FUNCTION test_avg AS 'org.apache.hadoop.hive.ql.udf.generic.GenericUDAFAverage';
sql/hive/src/test/resources/ql/src/test/queries/clientpositive/create_genericudaf.q:CREATE TEMPORARY FUNCTION test_avg AS 'org.apache.hadoop.hive.ql.udf.generic.GenericUDAFAverage';
sql/hive/src/test/resources/ql/src/test/queries/clientpositive/udaf_sum_list.q:-- GenericUDAFSumList has Converter which does not have default constructor
sql/hive/src/test/resources/ql/src/test/queries/clientpositive/udaf_sum_list.q:create temporary function sum_list as 'org.apache.hadoop.hive.ql.udf.generic.GenericUDAFSumList';
sql/hive/src/test/scala/org/apache/spark/sql/hive/execution/HiveSQLViewSuite.scala:    val permanentFuncClass =
sql/hive/src/test/scala/org/apache/spark/sql/hive/execution/HiveSQLViewSuite.scala:      sql(s"CREATE FUNCTION $permanentFuncName AS '$permanentFuncClass'")
sql/hive/src/test/scala/org/apache/spark/sql/hive/execution/ObjectHashAggregateSuite.scala:import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFMax
sql/hive/src/test/scala/org/apache/spark/sql/hive/execution/ObjectHashAggregateSuite.scala:    sql(s"CREATE TEMPORARY FUNCTION hive_max AS '${classOf[GenericUDAFMax].getName}'")
sql/hive/src/test/scala/org/apache/spark/sql/hive/execution/HiveUDAFSuite.scala:import org.apache.hadoop.hive.ql.udf.generic.{AbstractGenericUDAFResolver, GenericUDAFEvaluator, GenericUDAFMax}
sql/hive/src/test/scala/org/apache/spark/sql/hive/execution/HiveUDAFSuite.scala:import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator.{AggregationBuffer, Mode}
sql/hive/src/test/scala/org/apache/spark/sql/hive/execution/HiveUDAFSuite.scala:    sql(s"CREATE TEMPORARY FUNCTION hive_max AS '${classOf[GenericUDAFMax].getName}'")
sql/hive/src/test/scala/org/apache/spark/sql/hive/execution/HiveUDAFSuite.scala:class MockUDAF extends AbstractGenericUDAFResolver {
sql/hive/src/test/scala/org/apache/spark/sql/hive/execution/HiveUDAFSuite.scala:  override def getEvaluator(info: Array[TypeInfo]): GenericUDAFEvaluator = new MockUDAFEvaluator
sql/hive/src/test/scala/org/apache/spark/sql/hive/execution/HiveUDAFSuite.scala:  extends GenericUDAFEvaluator.AbstractAggregationBuffer {
sql/hive/src/test/scala/org/apache/spark/sql/hive/execution/HiveUDAFSuite.scala:class MockUDAFEvaluator extends GenericUDAFEvaluator {
sql/hive/src/test/scala/org/apache/spark/sql/hive/execution/HiveUDFSuite.scala:    sql(s"CREATE TEMPORARY FUNCTION test_avg AS '${classOf[GenericUDAFAverage].getName}'")
sql/hive/src/test/scala/org/apache/spark/sql/hive/execution/HiveUDFSuite.scala:      // AbstractGenericUDAFResolver
sql/hive/src/test/scala/org/apache/spark/sql/hive/execution/HiveUDFSuite.scala:      testErrorMsgForFunc("testUDAFAverage", classOf[GenericUDAFAverage].getName)
sql/hive/src/test/scala/org/apache/spark/sql/hive/execution/HiveUDFSuite.scala:      // AbstractGenericUDAFResolver
sql/hive/src/test/scala/org/apache/spark/sql/hive/execution/HiveUDFSuite.scala:      sql(s"CREATE FUNCTION test_avg AS '${classOf[GenericUDAFAverage].getName}'")
sql/hive/src/test/scala/org/apache/spark/sql/hive/execution/HiveUDFSuite.scala:            sql(s"CREATE FUNCTION dAtABaSe1.test_avg AS '${classOf[GenericUDAFAverage].getName}'")
sql/hive/src/test/scala/org/apache/spark/sql/execution/benchmark/ObjectHashAggregateExecBenchmark.scala:import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFPercentileApprox
sql/hive/src/test/scala/org/apache/spark/sql/execution/benchmark/ObjectHashAggregateExecBenchmark.scala:    registerHiveFunction("hive_percentile_approx", classOf[GenericUDAFPercentileApprox])
sql/hive/src/main/scala/org/apache/spark/sql/hive/hiveUDFs.scala:import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator.AggregationBuffer
sql/hive/src/main/scala/org/apache/spark/sql/hive/hiveUDFs.scala: *  1. An instance of some concrete `GenericUDAFEvaluator.AggregationBuffer` class
sql/hive/src/main/scala/org/apache/spark/sql/hive/hiveUDFs.scala: *     This is the native Hive representation of an aggregation state. Hive `GenericUDAFEvaluator`
sql/hive/src/main/scala/org/apache/spark/sql/hive/hiveUDFs.scala: *     `GenericUDAFEvaluator.init()` method.
sql/hive/src/main/scala/org/apache/spark/sql/hive/hiveUDFs.scala: *  - `GenericUDAFEvaluator.terminatePartial()`: from 2 to 3
sql/hive/src/main/scala/org/apache/spark/sql/hive/hiveUDFs.scala:  extends TypedImperativeAggregate[GenericUDAFEvaluator.AggregationBuffer]
sql/hive/src/main/scala/org/apache/spark/sql/hive/hiveUDFs.scala:  private def newEvaluator(): GenericUDAFEvaluator = {
sql/hive/src/main/scala/org/apache/spark/sql/hive/hiveUDFs.scala:      new GenericUDAFBridge(funcWrapper.createFunction[UDAF]())
sql/hive/src/main/scala/org/apache/spark/sql/hive/hiveUDFs.scala:      funcWrapper.createFunction[AbstractGenericUDAFResolver]()
sql/hive/src/main/scala/org/apache/spark/sql/hive/hiveUDFs.scala:    val parameterInfo = new SimpleGenericUDAFParameterInfo(inputInspectors, false, false)
sql/hive/src/main/scala/org/apache/spark/sql/hive/hiveUDFs.scala:    GenericUDAFEvaluator.Mode.PARTIAL1,
sql/hive/src/main/scala/org/apache/spark/sql/hive/hiveUDFs.scala:    evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL2, Array(partialResultInspector))
sql/hive/src/main/scala/org/apache/spark/sql/hive/hiveUDFs.scala:    GenericUDAFEvaluator.Mode.FINAL,
sql/hive/src/main/scala/org/apache/spark/sql/hive/hiveUDFs.scala:    // The 2nd argument of the Hive `GenericUDAFEvaluator.merge()` method is an input aggregation
sql/hive/src/main/scala/org/apache/spark/sql/hive/hiveUDFs.scala:    // calls `GenericUDAFEvaluator.terminatePartial()` to do the conversion.
sql/hive/src/main/scala/org/apache/spark/sql/hive/hiveUDFs.scala:      // `GenericUDAFEvaluator.terminatePartial()` converts an `AggregationBuffer` into an object
sql/hive/src/main/scala/org/apache/spark/sql/hive/hiveUDFs.scala:      // that can be inspected by the `ObjectInspector` returned by `GenericUDAFEvaluator.init()`.
sql/hive/src/main/scala/org/apache/spark/sql/hive/hiveUDFs.scala:      // `GenericUDAFEvaluator` doesn't provide any method that is capable to convert an object
sql/hive/src/main/scala/org/apache/spark/sql/hive/hiveUDFs.scala:      // returned by `GenericUDAFEvaluator.terminatePartial()` back to an `AggregationBuffer`. The
sql/hive/src/main/scala/org/apache/spark/sql/hive/HiveSessionCatalog.scala:import org.apache.hadoop.hive.ql.udf.generic.{AbstractGenericUDAFResolver, GenericUDF, GenericUDTF}
sql/hive/src/main/scala/org/apache/spark/sql/hive/HiveSessionCatalog.scala:        } else if (classOf[AbstractGenericUDAFResolver].isAssignableFrom(clazz)) {
sql/core/src/main/scala/org/apache/spark/sql/execution/objects.scala:    val (funcClass, methodName) = func match {
sql/core/src/main/scala/org/apache/spark/sql/execution/objects.scala:    val funcObj = Literal.create(func, ObjectType(funcClass))
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public void OpenSession(TOpenSessionReq req, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.OpenSession_call> resultHandler) throws org.apache.thrift.TException;
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public void CloseSession(TCloseSessionReq req, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.CloseSession_call> resultHandler) throws org.apache.thrift.TException;
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public void GetInfo(TGetInfoReq req, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.GetInfo_call> resultHandler) throws org.apache.thrift.TException;
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public void ExecuteStatement(TExecuteStatementReq req, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.ExecuteStatement_call> resultHandler) throws org.apache.thrift.TException;
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public void GetTypeInfo(TGetTypeInfoReq req, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.GetTypeInfo_call> resultHandler) throws org.apache.thrift.TException;
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public void GetCatalogs(TGetCatalogsReq req, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.GetCatalogs_call> resultHandler) throws org.apache.thrift.TException;
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public void GetSchemas(TGetSchemasReq req, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.GetSchemas_call> resultHandler) throws org.apache.thrift.TException;
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public void GetTables(TGetTablesReq req, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.GetTables_call> resultHandler) throws org.apache.thrift.TException;
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public void GetTableTypes(TGetTableTypesReq req, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.GetTableTypes_call> resultHandler) throws org.apache.thrift.TException;
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public void GetColumns(TGetColumnsReq req, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.GetColumns_call> resultHandler) throws org.apache.thrift.TException;
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public void GetFunctions(TGetFunctionsReq req, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.GetFunctions_call> resultHandler) throws org.apache.thrift.TException;
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public void GetOperationStatus(TGetOperationStatusReq req, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.GetOperationStatus_call> resultHandler) throws org.apache.thrift.TException;
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public void CancelOperation(TCancelOperationReq req, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.CancelOperation_call> resultHandler) throws org.apache.thrift.TException;
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public void CloseOperation(TCloseOperationReq req, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.CloseOperation_call> resultHandler) throws org.apache.thrift.TException;
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public void GetResultSetMetadata(TGetResultSetMetadataReq req, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.GetResultSetMetadata_call> resultHandler) throws org.apache.thrift.TException;
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public void FetchResults(TFetchResultsReq req, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.FetchResults_call> resultHandler) throws org.apache.thrift.TException;
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public void GetDelegationToken(TGetDelegationTokenReq req, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.GetDelegationToken_call> resultHandler) throws org.apache.thrift.TException;
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public void CancelDelegationToken(TCancelDelegationTokenReq req, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.CancelDelegationToken_call> resultHandler) throws org.apache.thrift.TException;
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public void RenewDelegationToken(TRenewDelegationTokenReq req, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.RenewDelegationToken_call> resultHandler) throws org.apache.thrift.TException;
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:  public static class AsyncClient extends org.apache.thrift.async.TAsyncClient implements AsyncIface {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public static class Factory implements org.apache.thrift.async.TAsyncClientFactory<AsyncClient> {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      private org.apache.thrift.async.TAsyncClientManager clientManager;
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public Factory(org.apache.thrift.async.TAsyncClientManager clientManager, org.apache.thrift.protocol.TProtocolFactory protocolFactory) {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public AsyncClient getAsyncClient(org.apache.thrift.transport.TNonblockingTransport transport) {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:        return new AsyncClient(protocolFactory, clientManager, transport);
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:    public AsyncClient(org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.async.TAsyncClientManager clientManager, org.apache.thrift.transport.TNonblockingTransport transport) {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public OpenSession_call(TOpenSessionReq req, org.apache.thrift.async.AsyncMethodCallback<OpenSession_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public CloseSession_call(TCloseSessionReq req, org.apache.thrift.async.AsyncMethodCallback<CloseSession_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public GetInfo_call(TGetInfoReq req, org.apache.thrift.async.AsyncMethodCallback<GetInfo_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public ExecuteStatement_call(TExecuteStatementReq req, org.apache.thrift.async.AsyncMethodCallback<ExecuteStatement_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public GetTypeInfo_call(TGetTypeInfoReq req, org.apache.thrift.async.AsyncMethodCallback<GetTypeInfo_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public GetCatalogs_call(TGetCatalogsReq req, org.apache.thrift.async.AsyncMethodCallback<GetCatalogs_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public GetSchemas_call(TGetSchemasReq req, org.apache.thrift.async.AsyncMethodCallback<GetSchemas_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public GetTables_call(TGetTablesReq req, org.apache.thrift.async.AsyncMethodCallback<GetTables_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public GetTableTypes_call(TGetTableTypesReq req, org.apache.thrift.async.AsyncMethodCallback<GetTableTypes_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public GetColumns_call(TGetColumnsReq req, org.apache.thrift.async.AsyncMethodCallback<GetColumns_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public GetFunctions_call(TGetFunctionsReq req, org.apache.thrift.async.AsyncMethodCallback<GetFunctions_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public GetOperationStatus_call(TGetOperationStatusReq req, org.apache.thrift.async.AsyncMethodCallback<GetOperationStatus_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public CancelOperation_call(TCancelOperationReq req, org.apache.thrift.async.AsyncMethodCallback<CancelOperation_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public CloseOperation_call(TCloseOperationReq req, org.apache.thrift.async.AsyncMethodCallback<CloseOperation_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public GetResultSetMetadata_call(TGetResultSetMetadataReq req, org.apache.thrift.async.AsyncMethodCallback<GetResultSetMetadata_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public FetchResults_call(TFetchResultsReq req, org.apache.thrift.async.AsyncMethodCallback<FetchResults_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public GetDelegationToken_call(TGetDelegationTokenReq req, org.apache.thrift.async.AsyncMethodCallback<GetDelegationToken_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public CancelDelegationToken_call(TCancelDelegationTokenReq req, org.apache.thrift.async.AsyncMethodCallback<CancelDelegationToken_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
sql/hive-thriftserver/src/gen/java/org/apache/hive/service/cli/thrift/TCLIService.java:      public RenewDelegationToken_call(TRenewDelegationTokenReq req, org.apache.thrift.async.AsyncMethodCallback<RenewDelegationToken_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
sql/catalyst/src/test/scala/org/apache/spark/sql/catalyst/catalog/ExternalCatalogSuite.scala:      CatalogFunction(FunctionIdentifier("func1", Some("db2")), funcClass,
sql/catalyst/src/test/scala/org/apache/spark/sql/catalyst/catalog/ExternalCatalogSuite.scala:    assert(catalog.getFunction("db2", "func1").className == funcClass)
sql/catalyst/src/test/scala/org/apache/spark/sql/catalyst/catalog/ExternalCatalogSuite.scala:    assert(catalog.getFunction("db2", newName).className == funcClass)
sql/catalyst/src/test/scala/org/apache/spark/sql/catalyst/catalog/ExternalCatalogSuite.scala:    assert(catalog.getFunction("db2", "func1").className == funcClass)
sql/catalyst/src/test/scala/org/apache/spark/sql/catalyst/catalog/ExternalCatalogSuite.scala:    val myNewFunc = catalog.getFunction("db2", "func1").copy(className = newFuncClass)
sql/catalyst/src/test/scala/org/apache/spark/sql/catalyst/catalog/ExternalCatalogSuite.scala:    assert(catalog.getFunction("db2", "func1").className == newFuncClass)
sql/catalyst/src/test/scala/org/apache/spark/sql/catalyst/catalog/ExternalCatalogSuite.scala:  lazy val funcClass = "org.apache.spark.myFunc"
sql/catalyst/src/test/scala/org/apache/spark/sql/catalyst/catalog/ExternalCatalogSuite.scala:  lazy val newFuncClass = "org.apache.spark.myNewFunc"
sql/catalyst/src/test/scala/org/apache/spark/sql/catalyst/catalog/ExternalCatalogSuite.scala:    CatalogFunction(FunctionIdentifier(name, database), funcClass, Seq.empty[FunctionResource])
sql/catalyst/src/test/scala/org/apache/spark/sql/catalyst/catalog/SessionCatalogSuite.scala:        CatalogFunction(FunctionIdentifier("func1", Some("db2")), funcClass,
sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/plans/logical/object.scala:    val (funcClass, methodName) = func match {
sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/plans/logical/object.scala:    val funcObj = Literal.create(func, ObjectType(funcClass))
sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/aggregate/collect.scala:    // See: org.apache.hadoop.hive.ql.udf.generic.GenericUDAFMkCollectionEvaluator
sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/ScalaUDF.scala:    val funcCls = function.getClass.getSimpleName
sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/ScalaUDF.scala:    s"Failed to execute user defined function($funcCls: ($inputTypes) => ${dataType.simpleString})"

```
