# https://github.com/freemed/freeshim

```console
shim-library/src/main/java/org/freemedsoftware/device/DosingPumpCommand.java:public enum DosingPumpCommand {
shim-library/src/main/java/org/freemedsoftware/device/DosingPumpCommand.java:	DosingPumpCommand(String txt) {
shim-library/src/main/java/org/freemedsoftware/device/ShimDeviceManager.java:		if (deviceInstance instanceof DosingPumpInterface) {
shim-library/src/main/java/org/freemedsoftware/device/ShimDeviceManager.java:		DosingPumpInterface sDevice = (DosingPumpInterface) deviceInstance;
shim-library/src/main/java/org/freemedsoftware/device/DosingPumpSerialInterface.java:public abstract class DosingPumpSerialInterface implements DosingPumpInterface {
shim-library/src/main/java/org/freemedsoftware/device/DosingPumpSerialInterface.java:	protected Logger log = Logger.getLogger(DosingPumpSerialInterface.class);
shim-library/src/main/java/org/freemedsoftware/device/DosingPumpInterface.java:public interface DosingPumpInterface extends DeviceInterface {
shim-drivers/shim-driver-dosing-scilog/src/main/java/org/freemedsoftware/device/impl/DosingPumpScilogShim.java:import org.freemedsoftware.device.DosingPumpSerialInterface;
shim-drivers/shim-driver-dosing-scilog/src/main/java/org/freemedsoftware/device/impl/DosingPumpScilogShim.java:public class DosingPumpScilogShim extends DosingPumpSerialInterface {
shim-drivers/shim-driver-dosing-scilog/src/main/java/org/freemedsoftware/device/impl/DosingPumpScilogShim.java:	protected Logger log = Logger.getLogger(DosingPumpScilogShim.class);
shim-drivers/shim-driver-dosing-scilog/src/main/java/org/freemedsoftware/device/impl/DosingPumpScilogShim.java:	public DosingPumpScilogShim() {
shim-drivers/shim-driver-dosing-scilog/src/main/java/org/freemedsoftware/device/impl/DosingPumpScilogShim.java:		setConfigName("org.freemedsoftware.device.impl.DosingPumpScilogShim");
shim-drivers/shim-driver-dosing-scilog/src/main/java/org/freemedsoftware/device/impl/DosingPumpScilogShim.java:						.get("org.freemedsoftware.device.impl.DosingPumpScilogShim.primePumpDuration"));
shim-drivers/shim-driver-dosing-scilog/src/main/java/org/freemedsoftware/device/impl/DosingPumpScilogShim.java:						.get("org.freemedsoftware.device.impl.DosingPumpScilogShim.reversePumpDuration"));
shim-webapp/src/main/webapp/admin/control.jsp:		if (s.getDosingPumpDeviceManager().getActive()) {
shim-webapp/src/main/webapp/admin/control.jsp:			s.getDosingPumpDeviceManager().close();
shim-webapp/src/main/webapp/admin/control.jsp:			s.getDosingPumpDeviceManager().init();
shim-webapp/src/main/webapp/admin/status.jsp:	ShimDeviceManager<DosingPumpInterface> dosingPumpDeviceManager = s.getDosingPumpDeviceManager();
shim-webapp/src/main/webapp/admin/status.jsp:		<% if (dosingPumpDeviceManager == null) { %>
shim-webapp/src/main/webapp/admin/status.jsp:		<td><%= dosingPumpDeviceManager.getClassName() %></td>
shim-webapp/src/main/webapp/admin/status.jsp:		<td><%= dosingPumpDeviceManager.getActive() %></td>
shim-webapp/src/main/webapp/WEB-INF/shim-default.properties:driver.dosingpump=org.freemedsoftware.device.impl.DosingPumpScilogShim
shim-webapp/src/main/webapp/WEB-INF/shim-default.properties:org.freemedsoftware.device.impl.DosingPumpScilogShim.enabled=false
shim-webapp/src/main/webapp/WEB-INF/shim-default.properties:org.freemedsoftware.device.impl.DosingPumpScilogShim.port=/dev/ttyS0
shim-webapp/src/main/webapp/WEB-INF/shim-default.properties:org.freemedsoftware.device.impl.DosingPumpScilogShim.baud=9600
shim-webapp/src/main/webapp/WEB-INF/shim-default.properties:org.freemedsoftware.device.impl.DosingPumpScilogShim.timeout=100
shim-webapp/src/main/webapp/WEB-INF/shim-default.properties:org.freemedsoftware.device.impl.DosingPumpScilogShim.reversePumpDuration=30
shim-webapp/src/main/webapp/WEB-INF/shim-default.properties:org.freemedsoftware.device.impl.DosingPumpScilogShim.primePumpDuration=90
shim-webapp/src/main/java/org/freemedsoftware/shim/MasterControlServlet.java:import org.freemedsoftware.device.DosingPumpInterface;
shim-webapp/src/main/java/org/freemedsoftware/shim/MasterControlServlet.java:	protected static ShimDeviceManager<DosingPumpInterface> dosingPumpDeviceManager = null;
shim-webapp/src/main/java/org/freemedsoftware/shim/MasterControlServlet.java:		String dosingPumpDriver = config.getString("driver.dosingpump");
shim-webapp/src/main/java/org/freemedsoftware/shim/MasterControlServlet.java:		if (dosingPumpDriver != null) {
shim-webapp/src/main/java/org/freemedsoftware/shim/MasterControlServlet.java:			logger.info("Initializing dosing pump driver " + dosingPumpDriver);
shim-webapp/src/main/java/org/freemedsoftware/shim/MasterControlServlet.java:				dosingPumpDeviceManager = new ShimDeviceManager<DosingPumpInterface>(
shim-webapp/src/main/java/org/freemedsoftware/shim/MasterControlServlet.java:						dosingPumpDriver);
shim-webapp/src/main/java/org/freemedsoftware/shim/MasterControlServlet.java:				dosingPumpDeviceManager.getDeviceInstance().configure(
shim-webapp/src/main/java/org/freemedsoftware/shim/MasterControlServlet.java:				if (dosingPumpDeviceManager == null) {
shim-webapp/src/main/java/org/freemedsoftware/shim/MasterControlServlet.java:				dosingPumpDeviceManager.init();
shim-webapp/src/main/java/org/freemedsoftware/shim/MasterControlServlet.java:	public static ShimDeviceManager<DosingPumpInterface> getDosingPumpDeviceManager() {
shim-webapp/src/main/java/org/freemedsoftware/shim/MasterControlServlet.java:		return dosingPumpDeviceManager;
shim-webapp/src/main/java/org/freemedsoftware/shim/MasterControlServlet.java:		if (dosingPumpDeviceManager != null) {
shim-webapp/src/main/java/org/freemedsoftware/shim/MasterControlServlet.java:				dosingPumpDeviceManager.close();
shim-webapp/src/main/java/org/freemedsoftware/shim/ShimServiceImpl.java:import org.freemedsoftware.device.DosingPumpCommand;
shim-webapp/src/main/java/org/freemedsoftware/shim/ShimServiceImpl.java:import org.freemedsoftware.device.DosingPumpInterface;
shim-webapp/src/main/java/org/freemedsoftware/shim/ShimServiceImpl.java:	public String requestDosingAction(DosingPumpCommand command, String param)
shim-webapp/src/main/java/org/freemedsoftware/shim/ShimServiceImpl.java:		ShimDeviceManager<DosingPumpInterface> manager = MasterControlServlet
shim-webapp/src/main/java/org/freemedsoftware/shim/ShimServiceImpl.java:				.getDosingPumpDeviceManager();
shim-webapp/src/main/java/org/freemedsoftware/shim/ShimServiceImpl.java:		if (command == DosingPumpCommand.DISPENSE) {
shim-webapp/src/main/java/org/freemedsoftware/shim/ShimServiceImpl.java:		if (command == DosingPumpCommand.GET_STATUS) {
shim-webapp/src/main/java/org/freemedsoftware/shim/ShimServiceImpl.java:		if (command == DosingPumpCommand.PRIME) {
shim-webapp/src/main/java/org/freemedsoftware/shim/ShimServiceImpl.java:		if (command == DosingPumpCommand.REVERSE) {
shim-webapp/src/main/java/org/freemedsoftware/shim/ShimService.java:import org.freemedsoftware.device.DosingPumpCommand;
shim-webapp/src/main/java/org/freemedsoftware/shim/ShimService.java:			@PathParam("command") @WebParam(name = "command") DosingPumpCommand command,

```
