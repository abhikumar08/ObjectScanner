import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.example.objectscanner"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "com.example.objectscanner"
        minSdk = 26
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        compose = true
    }
    androidResources {
        // Keep the model uncompressed so it can be memory-mapped straight from the APK.
        noCompress += "tflite"
    }
    packaging {
        jniLibs {
            // The QNN DSP/NPU loader opens its skel libraries from nativeLibraryDir, so they
            // must be extracted at install time.
            useLegacyPackaging = true
            // QNN's own GPU backend is never used: the LiteRT GPU delegate covers the GPU.
            excludes += "**/libQnnGpu.so"
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget = JvmTarget.JVM_17
    }
}

dependencies {
    val cameraX = "1.6.2"
    val lifecycle = "2.11.0"
    val qnn = "2.51.0"

    implementation(platform("androidx.compose:compose-bom:2026.09.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:$lifecycle")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:$lifecycle")

    implementation("androidx.camera:camera-core:$cameraX")
    implementation("androidx.camera:camera-camera2:$cameraX")
    implementation("androidx.camera:camera-lifecycle:$cameraX")
    implementation("androidx.camera:camera-view:$cameraX")

    // LiteRT runtime (org.tensorflow.lite.Interpreter). See ObjectDetectorHelper for why
    // the Task Library is not used.
    implementation("com.google.ai.edge.litert:litert:1.4.2")
    implementation("com.google.ai.edge.litert:litert-gpu:1.4.2")

    // Qualcomm AI Engine Direct (QNN) delegate for Snapdragon NPUs and DSPs, arm64 only.
    // Licensed under Qualcomm's own terms, not Apache 2.0. Keep both at the same version.
    implementation("com.qualcomm.qti:qnn-litert-delegate:$qnn")
    implementation("com.qualcomm.qti:qnn-runtime:$qnn")
}
