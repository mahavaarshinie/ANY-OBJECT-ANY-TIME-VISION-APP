# Any-Object Real-Time Industrial Anomaly Detection Pipeline

An enterprise-grade, standalone industrial quality control dashboard powered by a pure **Next.js 14 (TypeScript)** architecture and **ONNX Runtime Node**. This system acts as a high-speed inspection camera for automated manufacturing lines, instantly detecting structural anomalies, surface defects, and physical degradation on components without needing a human operator.

Unlike traditional computer vision applications that rely on massive training sets of specific defects, this pipeline uses an advanced **Unsupervised Distribution Alignment (Variance-Scaled PatchCore Proxy)** framework. It learns completely from perfection—requiring only 2 to 3 reference images of a flawless object to calibrate its decision boundaries on the fly.

---

## Key Features

* **Dynamic Target Memory Enrollment:** Zero hardcoded dataset reliance. Upload 2-5 reference snapshots of *any physical object* (e.g., microchips, bottles, precision components) to instantly generate a temporary coreset matrix in local RAM.
* **Pure JavaScript Execution Edge Engine:** Operates entirely offline with zero multi-server orchestration or external Python API bottlenecks. Feature tracing runs directly inside Serverless Next.js Node routes.
* **Statistical Variance-Scaled Filtering:** Implements an aggregate Mahalanobis-proxy distance metric. The system computes the precise standard deviation of every deep feature channel across the reference set, filtering out background lighting gradients and text prints (e.g., unique component serial numbers) while locking strictly onto structural physical defects.
* **Interactive Dark-Mode Telemetry Desk:** Tailwinds-built interface providing real-time evaluation metrics, live line snapshot viewports, continuous distance discrepancy tracking, and automated threshold limit scoring.

---

## Architectural Overview & Deep Learning Pipeline

```mermaid
graph TD
    A[Raw Evaluation Image Input] --> B[ImageNet CHW Matrix Normalization]
    B --> C[ONNX Runtime Layer-2 Feature Extraction]
    C --> D[512-Channel Spatial Feature Map]
    D --> E[Statistical Variance Matching Engine]
    E --> F{Within Calibrated Threshold?}
    F -- Yes --> G[NORMAL COMPLIANT Status]
    F -- No --> H[ANOMALY DETECTED Warning]

