# ARCHITECTURE.md

## 1. Model Selection Defense

I selected **Qwen2.5-VL-2B** as my Video-Language Model (VLM) for this project. My rationale includes:

- **Fast inference:** Its low latency allows me to process video frames almost in real-time.
- **Proven video support:** It efficiently handles temporal video features.
- **Unsloth-compatible:** It integrates seamlessly with my existing video analysis pipeline.
- **Extensive documentation:** It makes troubleshooting, parameter tuning, and optimization easier.

### VRAM Comparison

| Model             | VRAM Requirement | Notes                                  |
|------------------|----------------|----------------------------------------|
| Qwen2.5-VL-2B     | 5–7 GB         | Fits Kaggle/T4 GPU, most accessible option |
| Model_X           | 10–12 GB       | Exceeds my VRAM budget; slower inference |
| Model_Y           | 8–10 GB        | Slightly slower; limited documentation |

**Conclusion:**  
I chose Qwen2.5-VL-2B because it balances compute feasibility with video-processing capabilities while staying within the 5–7 GB VRAM budget of a standard T4 GPU.

---

## 2. Frame Sampling Rationale

I use **boundary-aware sampling** instead of uniform sampling.  

**Reason:**  
Uniform sampling might skip important frames at the start or end of short operations. Boundary-aware sampling ensures I capture **temporal boundaries**, which improves recognition for operations like “Pack” or “Tape,” where start and end frames are critical.

## 3. Failure Mode Analysis

**Most Confused Operation:** `"Tape"` misclassified as `"Pack"`  

**Hypothesis:**  
- **Visual similarity:** Both operations involve hand movements around boxes.  
- **Temporal ambiguity:** “Tape” is a short action that often overlaps with the early frames of “Pack.”  

**Training & Data Challenges:**  
- **Video extraction issues:** Some models did not support direct video input, so frames weren’t extracted properly.  
- **Limited samples:** Certain operations had very few labeled videos, making learning difficult.  
- **VRAM constraints:** Large video models required high memory, forcing smaller batch sizes or shorter clips, which sometimes reduced accuracy.

**Mitigation Ideas:**  
- I increased sampling density at operation boundaries to capture short actions better.  
- I pre-processed videos to ensure frames were correctly extracted before training.  
- I focused on models like **Qwen2.5-VL-2B**, which fit my GPU VRAM and were compatible with video input.  
- I augmented my dataset with additional examples of confusing operations.
