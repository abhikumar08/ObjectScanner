# LiteRT and the GPU delegate ship their own keep rules. Native method names are kept by
# proguard-android-optimize.txt.

# The QNN delegate's JNI code reads QnnDelegate.Options fields and enums by name, and its AAR
# ships no keep rules.
-keep class com.qualcomm.qti.** { *; }
