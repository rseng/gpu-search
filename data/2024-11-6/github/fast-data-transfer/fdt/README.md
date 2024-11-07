# https://github.com/fast-data-transfer/fdt

```console
src/lia/util/net/common/Config.java:        return this.sLocalAddresses == null ? LocalHost.getStringPublicIPs4() : this.sLocalAddresses;
src/lia/util/net/common/LocalHost.java:    static public String getStringPublicIPs4() {
src/lia/util/net/common/AbstractFDTCloseable.java:            this.setName(" AsyncCloseThread [ " + workingQueue.size() + " ]");
src/lia/util/net/common/AbstractFDTCloseable.java:                    this.setName(" AsyncCloseThread waiting to take wqSize: " + workingQueue.size());
src/lia/util/net/common/AbstractFDTCloseable.java:                    this.setName(" AsyncCloseThread CLOSING [ " + closeable + " ] wqSize: " + workingQueue.size());
src/lia/util/net/common/Utils.java:        logPublishDate(jsonObject);
src/lia/util/net/common/Utils.java:    private static String logPublishDate(JSONObject jsonObject) throws JSONException {
src/lia/util/net/copy/FDTWriterSession.java:        final Map<String, FileSession> preProcMap = (hasPreProc) ? new HashMap<String, FileSession>() : null;
src/lia/util/net/copy/FDTWriterSession.java:                preProcMap.put(fwsCopy.fileName(), fwsCopy);
src/lia/util/net/copy/FDTWriterSession.java:                bPreProccessing = doPreprocess(preProcessFilters, preProcMap);
src/lia/util/net/copy/FDTWriterSession.java:    private boolean doPreprocess(String[] preProcessFilters, Map<String, FileSession> preProcMap) throws Exception {
src/lia/util/net/copy/FDTWriterSession.java:            for (final FileSession fws : preProcMap.values()) {
src/lia/util/net/copy/FDTWriterSession.java:                    preProcMap.remove(fName);
src/lia/util/net/copy/FDTWriterSession.java:        final String[] fList = preProcMap.keySet().toArray(new String[preProcMap.size()]);
src/lia/util/net/copy/FDTWriterSession.java:        processorInfo.fileSessionMap = preProcMap;
src/lia/util/net/copy/FDTWriterSession.java:        for (final FileSession fs : preProcMap.values()) {
src/lia/util/net/copy/FDTWriterSession.java:            final HashMap<String, FileSession> preprocMap = new HashMap<String, FileSession>();
src/lia/util/net/copy/FDTWriterSession.java:                        preprocMap.put(fws.fileName(), fws);
src/lia/util/net/copy/FDTWriterSession.java:                    final String[] fList = preprocMap.keySet().toArray(new String[preprocMap.size()]);
src/lia/util/net/copy/FDTWriterSession.java:                    processorInfo.fileSessionMap = new HashMap<String, FileSession>(preprocMap);
src/lia/util/net/copy/FDTReaderSession.java:        final int avProcMax = Math.max(avProcProp, Utils.availableProcessors());
src/lia/util/net/copy/FDTReaderSession.java:        fileBlockQueue = new ArrayBlockingQueue<>(avProcMax * rMul);

```
