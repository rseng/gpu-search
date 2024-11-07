# https://github.com/gorpipe/gor

```console
model/src/main/java/org/gorpipe/gor/model/DriverBackedFileReader.java:        return ((StreamSource)source).openClosable();
model/src/main/java/org/gorpipe/gor/driver/providers/stream/sources/file/FileSource.java:    public InputStream openClosable() {
model/src/main/java/org/gorpipe/gor/driver/providers/stream/sources/wrappers/ExtendedRangeWrapper.java:    public InputStream openClosable() {
model/src/main/java/org/gorpipe/gor/driver/providers/stream/sources/wrappers/WrappedStreamSource.java:    public InputStream openClosable() {
model/src/main/java/org/gorpipe/gor/driver/providers/stream/sources/wrappers/WrappedStreamSource.java:        return getWrapped().openClosable();
model/src/main/java/org/gorpipe/gor/driver/providers/stream/sources/wrappers/RetryStreamSourceWrapper.java:    public InputStream openClosable() {
model/src/main/java/org/gorpipe/gor/driver/providers/stream/sources/wrappers/RetryStreamSourceWrapper.java:        return retry.perform(() -> wrapStream(super.openClosable(), 0, null));
model/src/main/java/org/gorpipe/gor/driver/providers/stream/sources/StreamSource.java:    default InputStream openClosable()  {

```
