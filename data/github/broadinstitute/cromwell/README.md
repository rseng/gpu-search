# https://github.com/broadinstitute/cromwell

```console
docs/RuntimeAttributes.md:- [gpuCount, gpuType, and nvidiaDriverVersion](#gpucount-gputype-and-nvidiadriverversion)
docs/RuntimeAttributes.md:### `gpuCount`, `gpuType`, and `nvidiaDriverVersion`
docs/RuntimeAttributes.md:Attach GPUs to the instance when running on the Pipelines API([GPU documentation](https://cloud.google.com/compute/docs/gpus/)).
docs/RuntimeAttributes.md:Make sure to choose a zone for which the type of GPU you want to attach is available.
docs/RuntimeAttributes.md:The types of compute GPU supported are:
docs/RuntimeAttributes.md:* `nvidia-tesla-k80` 
docs/RuntimeAttributes.md:* `nvidia-tesla-v100`
docs/RuntimeAttributes.md:* `nvidia-tesla-p100`
docs/RuntimeAttributes.md:* `nvidia-tesla-p4`
docs/RuntimeAttributes.md:* `nvidia-tesla-t4`
docs/RuntimeAttributes.md:For the latest list of supported GPU's, please visit [Google's GPU documentation](nvidia-drivers-us-public).
docs/RuntimeAttributes.md:The default driver is `418.87.00`, you may specify your own via the `nvidiaDriverVersion` key.  Make sure that driver exists in the `nvidia-drivers-us-public` beforehand, per the [Google Pipelines API documentation](https://cloud.google.com/genomics/reference/rest/Shared.Types/Metadata#VirtualMachine). 
docs/RuntimeAttributes.md:    gpuType: "nvidia-tesla-k80"
docs/RuntimeAttributes.md:    gpuCount: 2
docs/RuntimeAttributes.md:    nvidiaDriverVersion: "418.87.00"
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:import cromwell.backend.google.pipelines.common.GpuResource.GpuType
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:class PipelinesApiGpuAttributesSpec extends AnyWordSpecLike with Matchers with PipelinesApiRuntimeAttributesSpecsMixin {
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:  val validGpuTypes = List(
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:    (Option(WomString("nvidia-tesla-k80")), Option(GpuType.NVIDIATeslaK80)),
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:    (Option(WomString("nvidia-tesla-p100")), Option(GpuType.NVIDIATeslaP100)),
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:    (Option(WomString("custom-gpu-24601")), Option(GpuType("custom-gpu-24601"))),
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:  val invalidGpuTypes = List(WomSingleFile("nvidia-tesla-k80"), WomInteger(100))
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:  val validGpuCounts = List(
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:  val invalidGpuCounts = List(WomString("ten"), WomFloat(1.0))
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:  validGpuTypes foreach { case (validGpuType, expectedGpuTypeValue) =>
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:    validGpuCounts foreach { case (validGpuCount, expectedGpuCountValue) =>
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:      s"validate the valid gpu type '$validGpuType' and count '$validGpuCount'" in {
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:        ) ++ validGpuType.map(t => "gpuType" -> t) ++ validGpuCount.map(c => "gpuCount" -> c)
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:        expectedGpuTypeValue match {
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:          case Some(v) => actualRuntimeAttributes.gpuResource.exists(_.gpuType == v)
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:          case None => actualRuntimeAttributes.gpuResource.foreach(_.gpuType == GpuType.DefaultGpuType)
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:        expectedGpuCountValue match {
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:          case Some(v) => actualRuntimeAttributes.gpuResource.exists(_.gpuCount.value == v)
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:          case None => actualRuntimeAttributes.gpuResource.foreach(_.gpuCount.value == GpuType.DefaultGpuCount.value)
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:    invalidGpuCounts foreach { invalidGpuCount =>
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:      s"not validate a valid gpu type '$validGpuType' but an invalid gpu count '$invalidGpuCount'" in {
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:        ) ++ validGpuType.map(t => "gpuType" -> t) + ("gpuCount" -> invalidGpuCount)
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:                                                  s"Invalid gpu count. Expected positive Int but got"
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:  invalidGpuTypes foreach { invalidGpuType =>
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:    invalidGpuCounts foreach { invalidGpuCount =>
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:      s"not validate a invalid gpu type '$invalidGpuType' and invalid gpu count '$invalidGpuCount'" in {
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:        ) + ("gpuType" -> invalidGpuType) + ("gpuCount" -> invalidGpuCount)
supportedBackends/google/pipelines/common/src/test/scala/cromwell/backend/google/pipelines/common/PipelinesApiGpuAttributesSpec.scala:                                                  s"Invalid gpu count. Expected positive Int but got"
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/GpuTypeValidation.scala:import cromwell.backend.google.pipelines.common.GpuResource.GpuType
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/GpuTypeValidation.scala:object GpuTypeValidation {
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/GpuTypeValidation.scala:  lazy val instance: RuntimeAttributesValidation[GpuType] = new GpuTypeValidation
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/GpuTypeValidation.scala:  lazy val optional: OptionalRuntimeAttributesValidation[GpuType] = instance.optional
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/GpuTypeValidation.scala:class GpuTypeValidation extends RuntimeAttributesValidation[GpuType] {
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/GpuTypeValidation.scala:  override def key = RuntimeAttributesKeys.GpuTypeKey
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/GpuTypeValidation.scala:  override def validateValue: PartialFunction[WomValue, ErrorOr[GpuType]] = {
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/GpuTypeValidation.scala:    case WomString(s) => GpuType(s).validNel
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/GpuTypeValidation.scala:      s"Invalid '$key': String value required but got ${other.womType.friendlyName}. See ${GpuType.MoreDetailsURL} for a list of options".invalidNel
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:import cromwell.backend.google.pipelines.common.GpuResource.GpuType
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:object GpuResource {
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:  val DefaultNvidiaDriverVersion = "418.87.00"
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:  final case class GpuType(name: String) {
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:  object GpuType {
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:    val NVIDIATeslaP100 = GpuType("nvidia-tesla-p100")
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:    val NVIDIATeslaK80 = GpuType("nvidia-tesla-k80")
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:    val DefaultGpuType: GpuType = NVIDIATeslaK80
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:    val DefaultGpuCount: Int Refined Positive = refineMV[Positive](1)
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:    val MoreDetailsURL = "https://cloud.google.com/compute/docs/gpus/"
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:final case class GpuResource(gpuType: GpuType,
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:                             gpuCount: Int Refined Positive,
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:                             nvidiaDriverVersion: String = GpuResource.DefaultNvidiaDriverVersion
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:                                               gpuResource: Option[GpuResource],
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:  private def gpuTypeValidation(runtimeConfig: Option[Config]): OptionalRuntimeAttributesValidation[GpuType] =
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:    GpuTypeValidation.optional
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:  val GpuDriverVersionKey = "nvidiaDriverVersion"
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:  private def gpuDriverValidation(runtimeConfig: Option[Config]): OptionalRuntimeAttributesValidation[String] =
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:    new StringRuntimeAttributesValidation(GpuDriverVersionKey).optional
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:  private def gpuCountValidation(
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:  ): OptionalRuntimeAttributesValidation[Int Refined Positive] = GpuValidation.optional
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:        gpuCountValidation(runtimeConfig),
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:        gpuTypeValidation(runtimeConfig),
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:        gpuDriverValidation(runtimeConfig),
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:    // GPU
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:    lazy val gpuType: Option[GpuType] =
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:      RuntimeAttributesValidation.extractOption(gpuTypeValidation(runtimeAttrsConfig).key, validatedRuntimeAttributes)
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:    lazy val gpuCount: Option[Int Refined Positive] =
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:      RuntimeAttributesValidation.extractOption(gpuCountValidation(runtimeAttrsConfig).key, validatedRuntimeAttributes)
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:    lazy val gpuDriver: Option[String] =
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:      RuntimeAttributesValidation.extractOption(gpuDriverValidation(runtimeAttrsConfig).key, validatedRuntimeAttributes)
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:    val gpuResource: Option[GpuResource] = if (gpuType.isDefined || gpuCount.isDefined || gpuDriver.isDefined) {
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:        GpuResource(gpuType.getOrElse(GpuType.DefaultGpuType),
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:                    gpuCount.getOrElse(GpuType.DefaultGpuCount),
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:                    gpuDriver.getOrElse(GpuResource.DefaultNvidiaDriverVersion)
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/PipelinesApiRuntimeAttributes.scala:      gpuResource,
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/GpuValidation.scala:import wom.RuntimeAttributesKeys.GpuKey
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/GpuValidation.scala:object GpuValidation {
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/GpuValidation.scala:  lazy val instance: RuntimeAttributesValidation[Int Refined Positive] = new GpuValidation(GpuKey)
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/GpuValidation.scala:class GpuValidation(attributeName: String) extends PositiveIntRuntimeAttributesValidation(attributeName) {
supportedBackends/google/pipelines/common/src/main/scala/cromwell/backend/google/pipelines/common/GpuValidation.scala:      s"Invalid gpu count. Expected positive Int but got ${other.womType.friendlyName} ${other.toWomString}".invalidNel
supportedBackends/google/pipelines/v2beta/src/test/scala/cromwell/backend/google/pipelines/v2beta/api/request/GetRequestHandlerSpec.scala:      |          "nvidiaDriverVersion": "450.51.06",
supportedBackends/google/pipelines/v2beta/src/test/scala/cromwell/backend/google/pipelines/v2beta/api/request/GetRequestHandlerSpec.scala:      |          "nvidiaDriverVersion": "450.51.06",
supportedBackends/google/pipelines/v2beta/src/test/scala/cromwell/backend/google/pipelines/v2beta/api/request/GetRequestHandlerSpec.scala:       |          "nvidiaDriverVersion": "",
supportedBackends/google/pipelines/v2beta/src/test/scala/cromwell/backend/google/pipelines/v2beta/api/request/GetRequestHandlerSpec.scala:       |          "nvidiaDriverVersion": "",
supportedBackends/google/pipelines/v2beta/src/test/scala/cromwell/backend/google/pipelines/v2beta/api/request/GetRequestHandlerSpec.scala:       |          "nvidiaDriverVersion": "450.51.06",
supportedBackends/google/pipelines/v2beta/src/main/scala/cromwell/backend/google/pipelines/v2beta/api/request/RunRequestHandler.scala:          Option("See https://cloud.google.com/compute/docs/gpus/ for a list of supported accelerators.")
supportedBackends/google/pipelines/v2beta/src/main/scala/cromwell/backend/google/pipelines/v2beta/PipelinesUtilityConversions.scala:import cromwell.backend.google.pipelines.common.{GpuResource, MachineConstraints, PipelinesApiRuntimeAttributes}
supportedBackends/google/pipelines/v2beta/src/main/scala/cromwell/backend/google/pipelines/v2beta/PipelinesUtilityConversions.scala:  def toAccelerator(gpuResource: GpuResource): Accelerator =
supportedBackends/google/pipelines/v2beta/src/main/scala/cromwell/backend/google/pipelines/v2beta/PipelinesUtilityConversions.scala:    new Accelerator().setCount(gpuResource.gpuCount.value.toLong).setType(gpuResource.gpuType.toString)
supportedBackends/google/pipelines/v2beta/src/main/scala/cromwell/backend/google/pipelines/v2beta/LifeSciencesFactory.scala:      val accelerators = createPipelineParameters.runtimeAttributes.gpuResource.map(toAccelerator).toList.asJava
supportedBackends/google/pipelines/v2beta/src/main/scala/cromwell/backend/google/pipelines/v2beta/LifeSciencesFactory.scala:      createPipelineParameters.runtimeAttributes.gpuResource foreach { resource =>
supportedBackends/google/pipelines/v2beta/src/main/scala/cromwell/backend/google/pipelines/v2beta/LifeSciencesFactory.scala:        virtualMachine.setNvidiaDriverVersion(resource.nvidiaDriverVersion)
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributesSpec.scala:    gpuResource = None,
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:import cromwell.backend.google.batch.models.GpuResource.GpuType
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:class GcpBatchGpuAttributesSpec extends AnyWordSpecLike with Matchers with GcpBatchRuntimeAttributesSpecsMixin {
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:  val validGpuTypes = List(
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:    (Option(WomString("nvidia-tesla-k80")), Option(GpuType.NVIDIATeslaK80)),
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:    (Option(WomString("nvidia-tesla-p100")), Option(GpuType.NVIDIATeslaP100)),
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:    (Option(WomString("custom-gpu-24601")), Option(GpuType("custom-gpu-24601"))),
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:  val invalidGpuTypes = List(WomSingleFile("nvidia-tesla-k80"), WomInteger(100))
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:  val validGpuCounts = List(
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:  val invalidGpuCounts = List(WomString("ten"), WomFloat(1.0))
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:  validGpuTypes foreach { case (validGpuType, expectedGpuTypeValue) =>
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:    validGpuCounts foreach { case (validGpuCount, expectedGpuCountValue) =>
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:      s"validate the valid gpu type '$validGpuType' and count '$validGpuCount'" in {
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:        ) ++ validGpuType.map(t => "gpuType" -> t) ++ validGpuCount.map(c => "gpuCount" -> c)
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:        expectedGpuTypeValue match {
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:          case Some(v) => actualRuntimeAttributes.gpuResource.exists(_.gpuType == v)
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:          case None => actualRuntimeAttributes.gpuResource.foreach(_.gpuType == GpuType.DefaultGpuType)
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:        expectedGpuCountValue match {
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:          case Some(v) => actualRuntimeAttributes.gpuResource.exists(_.gpuCount.value == v)
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:          case None => actualRuntimeAttributes.gpuResource.foreach(_.gpuCount.value == GpuType.DefaultGpuCount.value)
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:    invalidGpuCounts foreach { invalidGpuCount =>
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:      s"not validate a valid gpu type '$validGpuType' but an invalid gpu count '$invalidGpuCount'" in {
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:        ) ++ validGpuType.map(t => "gpuType" -> t) + ("gpuCount" -> invalidGpuCount)
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:                                                   s"Invalid gpu count. Expected positive Int but got"
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:  invalidGpuTypes foreach { invalidGpuType =>
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:    invalidGpuCounts foreach { invalidGpuCount =>
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:      s"not validate a invalid gpu type '$invalidGpuType' and invalid gpu count '$invalidGpuCount'" in {
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:        ) + ("gpuType" -> invalidGpuType) + ("gpuCount" -> invalidGpuCount)
supportedBackends/google/batch/src/test/scala/cromwell/backend/google/batch/models/GcpBatchAttributeSpec.scala:                                                   s"Invalid gpu count. Expected positive Int but got"
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/api/GcpBatchRequestFactoryImpl.scala:    // set GPU count to 0 if not included in workflow
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/api/GcpBatchRequestFactoryImpl.scala:    val gpuAccelerators = accelerators.getOrElse(Accelerator.newBuilder.setCount(0).setType("")) // TODO: Driver version
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/api/GcpBatchRequestFactoryImpl.scala:    // add GPUs if GPU count is greater than 1
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/api/GcpBatchRequestFactoryImpl.scala:    if (gpuAccelerators.getCount >= 1) {
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/api/GcpBatchRequestFactoryImpl.scala:      val instancePolicyGpu = instancePolicy.toBuilder
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/api/GcpBatchRequestFactoryImpl.scala:      instancePolicyGpu.addAccelerators(gpuAccelerators).build
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/api/GcpBatchRequestFactoryImpl.scala:      instancePolicyGpu
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/api/GcpBatchRequestFactoryImpl.scala:    val gpuAccelerators = accelerators.getOrElse(Accelerator.newBuilder.setCount(0).setType(""))
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/api/GcpBatchRequestFactoryImpl.scala:    // add GPUs if GPU count is greater than or equal to 1
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/api/GcpBatchRequestFactoryImpl.scala:    if (gpuAccelerators.getCount >= 1) {
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/api/GcpBatchRequestFactoryImpl.scala:        InstancePolicyOrTemplate.newBuilder.setPolicy(instancePolicy).setInstallGpuDrivers(true).build
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/api/GcpBatchRequestFactoryImpl.scala:    // Set GPU accelerators
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/api/GcpBatchRequestFactoryImpl.scala:    val accelerators = runtimeAttributes.gpuResource.map(toAccelerator)
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:import cromwell.backend.google.batch.models.GpuResource.GpuType
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:import cromwell.backend.google.batch.util.{GpuTypeValidation, GpuValidation}
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:object GpuResource {
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:  val DefaultNvidiaDriverVersion = "418.87.00"
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:  final case class GpuType(name: String) {
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:  object GpuType {
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:    val NVIDIATeslaP100 = GpuType("nvidia-tesla-p100")
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:    val NVIDIATeslaK80 = GpuType("nvidia-tesla-k80")
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:    val DefaultGpuType: GpuType = NVIDIATeslaK80
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:    val DefaultGpuCount: Int Refined Positive = refineMV[Positive](1)
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:    val MoreDetailsURL = "https://cloud.google.com/compute/docs/gpus/"
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:final case class GpuResource(gpuType: GpuType, gpuCount: Int Refined Positive)
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:                                           gpuResource: Option[GpuResource],
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:  private def gpuTypeValidation(runtimeConfig: Option[Config]): OptionalRuntimeAttributesValidation[GpuType] =
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:    GpuTypeValidation.optional
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:  val GpuDriverVersionKey = "nvidiaDriverVersion"
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:  private def gpuDriverValidation(runtimeConfig: Option[Config]): OptionalRuntimeAttributesValidation[String] =
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:    new StringRuntimeAttributesValidation(GpuDriverVersionKey).optional
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:  private def gpuCountValidation(
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:  ): OptionalRuntimeAttributesValidation[Int Refined Positive] = GpuValidation.optional
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:        gpuCountValidation(runtimeConfig),
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:        gpuTypeValidation(runtimeConfig),
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:        gpuDriverValidation(runtimeConfig),
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:    // GPU
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:    lazy val gpuType: Option[GpuType] = RuntimeAttributesValidation
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:      .extractOption(gpuTypeValidation(runtimeAttrsConfig).key, validatedRuntimeAttributes)
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:    lazy val gpuCount: Option[Int Refined Positive] = RuntimeAttributesValidation
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:      .extractOption(gpuCountValidation(runtimeAttrsConfig).key, validatedRuntimeAttributes)
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:    lazy val gpuDriver: Option[String] =
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:      RuntimeAttributesValidation.extractOption(gpuDriverValidation(runtimeAttrsConfig).key, validatedRuntimeAttributes)
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:    val gpuResource: Option[GpuResource] = if (gpuType.isDefined || gpuCount.isDefined || gpuDriver.isDefined) {
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:        GpuResource(gpuType.getOrElse(GpuType.DefaultGpuType),
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:                    gpuCount
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:                      .getOrElse(GpuType.DefaultGpuCount)
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/models/GcpBatchRuntimeAttributes.scala:      gpuResource = gpuResource,
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/util/GpuTypeValidation.scala:import cromwell.backend.google.batch.models.GpuResource.GpuType
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/util/GpuTypeValidation.scala:object GpuTypeValidation {
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/util/GpuTypeValidation.scala:  lazy val instance: RuntimeAttributesValidation[GpuType] = new GpuTypeValidation
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/util/GpuTypeValidation.scala:  lazy val optional: OptionalRuntimeAttributesValidation[GpuType] = instance.optional
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/util/GpuTypeValidation.scala:class GpuTypeValidation extends RuntimeAttributesValidation[GpuType] {
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/util/GpuTypeValidation.scala:  override def key = RuntimeAttributesKeys.GpuTypeKey
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/util/GpuTypeValidation.scala:  override def validateValue: PartialFunction[WomValue, ErrorOr[GpuType]] = {
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/util/GpuTypeValidation.scala:    case WomString(s) => GpuType(s).validNel
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/util/GpuTypeValidation.scala:      s"Invalid '$key': String value required but got ${other.womType.friendlyName}. See ${GpuType.MoreDetailsURL} for a list of options".invalidNel
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/util/GpuValidation.scala:import wom.RuntimeAttributesKeys.GpuKey
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/util/GpuValidation.scala:object GpuValidation {
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/util/GpuValidation.scala:  lazy val instance: RuntimeAttributesValidation[Int Refined Positive] = new GpuValidation(GpuKey)
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/util/GpuValidation.scala:class GpuValidation(attributeName: String) extends PositiveIntRuntimeAttributesValidation(attributeName) {
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/util/GpuValidation.scala:      s"Invalid gpu count. Expected positive Int but got ${other.womType.friendlyName} ${other.toWomString}".invalidNel
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/util/BatchUtilityConversions.scala:import cromwell.backend.google.batch.models.{GcpBatchRuntimeAttributes, GpuResource}
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/util/BatchUtilityConversions.scala:  // Create accelerators for GPUs
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/util/BatchUtilityConversions.scala:  def toAccelerator(gpuResource: GpuResource): Accelerator.Builder =
supportedBackends/google/batch/src/main/scala/cromwell/backend/google/batch/util/BatchUtilityConversions.scala:    Accelerator.newBuilder.setCount(gpuResource.gpuCount.value.toLong).setType(gpuResource.gpuType.toString)
supportedBackends/sfs/src/test/scala/cromwell/backend/sfs/TestLocalAsyncJobExecutionActor.scala:    val asyncClass = classOf[TestLocalAsyncJobExecutionActor]
supportedBackends/sfs/src/test/scala/cromwell/backend/sfs/TestLocalAsyncJobExecutionActor.scala:      asyncJobExecutionActorClass = asyncClass,
wom/src/main/scala/wom/RuntimeAttributes.scala:  val GpuKey = "gpuCount"
wom/src/main/scala/wom/RuntimeAttributes.scala:  val GpuTypeKey = "gpuType"
backend/src/main/scala/cromwell/backend/BackendLifecycleActorFactory.scala:  def nameForCallCachingPurposes: String =
centaur/src/main/resources/standardTestCases/gpu_on_papi/valid.inputs.json:  "gpu_on_papi.gpus": [ "nvidia-tesla-t4", "nvidia-tesla-p100", "nvidia-tesla-p4" ]
centaur/src/main/resources/standardTestCases/gpu_on_papi/gpu_on_papi.wdl:workflow gpu_on_papi {
centaur/src/main/resources/standardTestCases/gpu_on_papi/gpu_on_papi.wdl:    Array[String] gpus
centaur/src/main/resources/standardTestCases/gpu_on_papi/gpu_on_papi.wdl:  scatter (gpu in gpus) {
centaur/src/main/resources/standardTestCases/gpu_on_papi/gpu_on_papi.wdl:    call task_with_gpu { input: gpuTypeInput = gpu }
centaur/src/main/resources/standardTestCases/gpu_on_papi/gpu_on_papi.wdl:    Array[Int] reported_gpu_counts = task_with_gpu.gpuCount
centaur/src/main/resources/standardTestCases/gpu_on_papi/gpu_on_papi.wdl:    Array[String] reported_gpu_types = task_with_gpu.gpuType
centaur/src/main/resources/standardTestCases/gpu_on_papi/gpu_on_papi.wdl:task task_with_gpu {
centaur/src/main/resources/standardTestCases/gpu_on_papi/gpu_on_papi.wdl:    String gpuTypeInput
centaur/src/main/resources/standardTestCases/gpu_on_papi/gpu_on_papi.wdl:    Int gpuCount = metadata.guestAccelerators[0].acceleratorCount
centaur/src/main/resources/standardTestCases/gpu_on_papi/gpu_on_papi.wdl:    String gpuType = metadata.guestAccelerators[0].acceleratorType
centaur/src/main/resources/standardTestCases/gpu_on_papi/gpu_on_papi.wdl:    gpuCount: 1
centaur/src/main/resources/standardTestCases/gpu_on_papi/gpu_on_papi.wdl:    gpuType: gpuTypeInput
centaur/src/main/resources/standardTestCases/gpu_on_papi/gpu_cuda_image.wdl:workflow gpu_cuda_image {
centaur/src/main/resources/standardTestCases/gpu_on_papi/gpu_cuda_image.wdl:		nvidia-modprobe --version > modprobe
centaur/src/main/resources/standardTestCases/gpu_on_papi/gpu_cuda_image.wdl:		nvidia-smi > smi
centaur/src/main/resources/standardTestCases/gpu_on_papi/gpu_cuda_image.wdl:        docker: "nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04"
centaur/src/main/resources/standardTestCases/gpu_on_papi/gpu_cuda_image.wdl:        gpuType: "nvidia-tesla-k80"
centaur/src/main/resources/standardTestCases/gpu_on_papi/gpu_cuda_image.wdl:        gpuCount: 1
centaur/src/main/resources/standardTestCases/gpu_on_papi/gpu_cuda_image.wdl:        nvidiaDriverVersion: driver_version
centaur/src/main/resources/standardTestCases/gpu_on_papi/invalid.inputs.json:  "gpu_on_papi.gpus": [ "nonsense value" ]
centaur/src/main/resources/standardTestCases/gpu_on_papi_valid.test:name: gpu_on_papi_valid
centaur/src/main/resources/standardTestCases/gpu_on_papi_valid.test:  workflow: gpu_on_papi/gpu_on_papi.wdl
centaur/src/main/resources/standardTestCases/gpu_on_papi_valid.test:  inputs: gpu_on_papi/valid.inputs.json
centaur/src/main/resources/standardTestCases/gpu_on_papi_valid.test:  "outputs.gpu_on_papi.reported_gpu_counts.0": 1
centaur/src/main/resources/standardTestCases/gpu_on_papi_valid.test:  "outputs.gpu_on_papi.reported_gpu_counts.1": 1
centaur/src/main/resources/standardTestCases/gpu_on_papi_valid.test:  "outputs.gpu_on_papi.reported_gpu_counts.2": 1
centaur/src/main/resources/standardTestCases/gpu_on_papi_valid.test:  "outputs.gpu_on_papi.reported_gpu_types.0": "https://www.googleapis.com/compute/v1/projects/broad-dsde-cromwell-dev/zones/us-central1-c/acceleratorTypes/nvidia-tesla-t4"
centaur/src/main/resources/standardTestCases/gpu_on_papi_valid.test:  "outputs.gpu_on_papi.reported_gpu_types.1": "https://www.googleapis.com/compute/v1/projects/broad-dsde-cromwell-dev/zones/us-central1-c/acceleratorTypes/nvidia-tesla-p100"
centaur/src/main/resources/standardTestCases/gpu_on_papi_valid.test:  "outputs.gpu_on_papi.reported_gpu_types.2": "https://www.googleapis.com/compute/v1/projects/broad-dsde-cromwell-dev/zones/us-central1-c/acceleratorTypes/nvidia-tesla-p4"
centaur/src/main/resources/standardTestCases/gpu_on_papi_invalid.test:name: gpu_on_papi_invalid
centaur/src/main/resources/standardTestCases/gpu_on_papi_invalid.test:  workflow: gpu_on_papi/gpu_on_papi.wdl
centaur/src/main/resources/standardTestCases/gpu_on_papi_invalid.test:  inputs: gpu_on_papi/invalid.inputs.json
centaur/src/main/resources/standardTestCases/gpu_on_papi_invalid.test:  "calls.gpu_on_papi.task_with_gpu.failures.0.message": "Unable to complete PAPI request due to a problem with the request (Error: validating pipeline: unsupported accelerator: \"nonsense value\"). See https://cloud.google.com/compute/docs/gpus/ for a list of supported accelerators."
centaur/src/main/resources/standardTestCases/gpu_cuda_image.test:name: gpu_cuda_image
centaur/src/main/resources/standardTestCases/gpu_cuda_image.test:  workflow: gpu_on_papi/gpu_cuda_image.wdl
centaur/src/main/resources/standardTestCases/gpu_cuda_image.test:  "outputs.gpu_cuda_image.modprobe_check.0": "good"
centaur/src/main/resources/standardTestCases/gpu_cuda_image.test:  "outputs.gpu_cuda_image.smi_check.0": "good"
centaur/src/main/resources/standardTestCases/gcpbatch_gpu_on_papi_invalid.test:name: gcpbatch_gpu_on_papi_invalid
centaur/src/main/resources/standardTestCases/gcpbatch_gpu_on_papi_invalid.test:  workflow: gpu_on_papi/gpu_on_papi.wdl
centaur/src/main/resources/standardTestCases/gcpbatch_gpu_on_papi_invalid.test:  inputs: gpu_on_papi/invalid.inputs.json
centaur/src/main/resources/standardTestCases/gcpbatch_gpu_on_papi_invalid.test:  "calls.gpu_on_papi.task_with_gpu.failures.0.message": "Unable to complete Batch request due to a problem with the request (io.grpc.StatusRuntimeException: INVALID_ARGUMENT: Accelerator field is invalid. Accelerator with type nonsense value is not supported for Batch now. Please make sure that the type is lowercase formatted as name field in command `gcloud compute accelerator-types list` shows if it exists in the list.). "
CHANGELOG.md: * Added Nvidia driver install (default 418) ([#7235](https://github.com/broadinstitute/cromwell/pull/7235}))
CHANGELOG.md:### Nvidia GPU Driver Update
CHANGELOG.md:The default driver for Nvidia GPU's on Google Cloud has been updated from `390` to `418.87.00`.  A user may override this option at anytime by providing the `nvidiaDriverVersion` runtime attribute.  See the [Runtime Attribute description for GPUs](https://cromwell.readthedocs.io/en/stable/RuntimeAttributes/#runtime-attribute-descriptions) for detailed information.
CHANGELOG.md:#### nVidia Driver Attribute Change
CHANGELOG.md:The runtime attribute `nvidia-driver-version` was previously allowed only as a default runtime attribute in configuration.
CHANGELOG.md:Because WDL does not allow attribute names to contain `-` characters, this has been changed to `nvidiaDriverVersion`.
CHANGELOG.md:#### GPU Attributes
CHANGELOG.md:* The `gpuType` attribute is no longer validated against a whitelist at workflow submission time. Instead, validation now happens at runtime. This allows any valid accelerator to be used.
CHANGELOG.md:* The `nvidiaDriverVersion` attribute is now available in WDL `runtime` sections. The default continues to be `390.46` which applies if and only if GPUs are being used.
CHANGELOG.md:* A default `gpuType` ("nvidia-tesla-k80") will now be applied if `gpuCount` is specified but `gpuType` is not.
CHANGELOG.md:* Similarly, a default `gpuCount` (1) will be applied if `gpuType` is specified but `cpuCount` is not.
CHANGELOG.md:### GPU
CHANGELOG.md:The PAPI backend now supports specifying GPU through WDL runtime attributes:
CHANGELOG.md:    gpuType: "nvidia-tesla-k80"
CHANGELOG.md:    gpuCount: 2
CHANGELOG.md:The two types of GPU supported are `nvidia-tesla-k80` and `nvidia-tesla-p100`
CHANGELOG.md:**Important**: Before adding a GPU, make sure it is available in the zone the job is running in: https://cloud.google.com/compute/docs/gpus/
engine/src/test/scala/cromwell/engine/workflow/lifecycle/execution/callcaching/EngineJobHashingActorSpec.scala:        backendNameForCallCachingPurposes = backendName,
engine/src/main/scala/cromwell/engine/workflow/lifecycle/execution/callcaching/CallCacheHashingJobActor.scala:                               backendNameForCallCachingPurposes: String,
engine/src/main/scala/cromwell/engine/workflow/lifecycle/execution/callcaching/CallCacheHashingJobActor.scala:    val backendNameHash = HashResult(HashKey("backend name"), backendNameForCallCachingPurposes.md5HashValue)
engine/src/main/scala/cromwell/engine/workflow/lifecycle/execution/callcaching/CallCacheHashingJobActor.scala:            backendNameForCallCachingPurposes: String,
engine/src/main/scala/cromwell/engine/workflow/lifecycle/execution/callcaching/CallCacheHashingJobActor.scala:      backendNameForCallCachingPurposes,
engine/src/main/scala/cromwell/engine/workflow/lifecycle/execution/callcaching/EngineJobHashingActor.scala:                            backendNameForCallCachingPurposes: String,
engine/src/main/scala/cromwell/engine/workflow/lifecycle/execution/callcaching/EngineJobHashingActor.scala:        backendNameForCallCachingPurposes,
engine/src/main/scala/cromwell/engine/workflow/lifecycle/execution/callcaching/EngineJobHashingActor.scala:            backendNameForCallCachingPurposes: String,
engine/src/main/scala/cromwell/engine/workflow/lifecycle/execution/callcaching/EngineJobHashingActor.scala:      backendNameForCallCachingPurposes = backendNameForCallCachingPurposes,
engine/src/main/scala/cromwell/engine/workflow/lifecycle/execution/job/EngineJobExecutionActor.scala:          backendLifecycleActorFactory.nameForCallCachingPurposes,

```
