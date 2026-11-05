**Volume 39. Hardware for AI**

# 01. GPU Architecture

## 01.00 Overview

![](images/image1.png){width="7.268055555555556in" height="7.268055555555556in"}

그래픽 처리 장치(Graphics Processing Unit, GPU)는 전문적인 그래픽 렌더링(Graphics Rendering) 장치에서 인공지능(Artificial Intelligence, AI)을 위한 핵심 컴퓨팅 플랫폼(Computing Platform)으로 발전해 왔다. 대규모 병렬 아키텍처(Massively Parallel Architecture)는 수천 개의 산술 연산(Arithmetic Operations)을 동시에 실행할 수 있도록 지원하며, 신경망 학습(Neural Network Training), 추론(Inference), 과학 시뮬레이션(Scientific Simulation), 대규모 수치 계산(Numerical Computation)에 특히 효과적이다. 효율적인 인공지능 시스템을 설계하려면 GPU 아키텍처에 대한 이해가 필수적이다.

중앙 처리 장치(Central Processing Unit, CPU)는 낮은 지연시간(Low Latency), 복잡한 명령어 처리(Instruction Handling), 순차 실행 성능(Sequential Performance)을 우선시하는 반면, GPU는 연산 처리량(Computational Throughput)을 강조한다. GPU는 대규모 작업 부하(Workload)를 다수의 실행 유닛(Execution Unit)에 분산하여 여러 데이터 요소를 동시에 처리한다. 이러한 아키텍처 차이로 인해 GPU는 행렬 곱셈(Matrix Multiplication), 합성곱(Convolution), 텐서 연산(Tensor Operations) 및 반복적인 수학 연산에 특히 적합하다.

현대 GPU는 여러 개의 스트리밍 멀티프로세서(Streaming Multiprocessor, SM)로 구성되며, 각 SM은 병렬 스레드(Thread) 그룹의 실행을 담당한다. SM은 산술 실행 유닛(Arithmetic Execution Unit), 스케줄링 메커니즘(Scheduling Mechanism), 레지스터(Register), 설정 가능한 온칩 메모리(On-Chip Memory) 자원을 통합한다. GPU는 스레드 블록(Thread Block)을 여러 SM에 분산하고 실행 가능한 워프(Warp)를 빠르게 전환함으로써 메모리 접근 지연시간을 은닉(Latency Hiding)하면서 높은 연산 활용률을 유지한다.

GPU 실행은 일반적으로 단일 명령어 다중 스레드(Single Instruction, Multiple Threads, SIMT) 프로그래밍 모델(Programming Model)을 따른다. 스레드는 워프(Warp) 단위로 구성되며, NVIDIA 아키텍처에서는 일반적으로 32개의 스레드가 하나의 워프를 형성한다. 각 스레드는 독립적인 레지스터와 실행 상태(Execution State)를 유지하지만, 워프 내부에서 서로 다른 제어 흐름(Control Flow)이 발생하면 분기 경로를 별도로 처리해야 하므로 실행 효율이 감소할 수 있다.

CUDA 프로그래밍 모델(CUDA Programming Model)은 NVIDIA GPU의 병렬 처리 기능에 접근하기 위한 소프트웨어 추상화(Software Abstraction)를 제공한다. 개발자는 여러 스레드 블록으로 구성된 그리드(Grid)에서 실행되는 커널(Kernel)을 정의한다. 스레드 인덱싱(Thread Indexing)은 각 스레드가 처리할 데이터 요소를 결정하며, 동기화(Synchronization)와 메모리 관리(Memory Management)는 연산을 조정한다. CUDA는 모든 하드웨어 명령어를 직접 제어하지 않고도 GPU 자원을 활용할 수 있도록 한다.

텐서 코어(Tensor Core)는 인공지능 가속(AI Acceleration)을 위한 중요한 아키텍처 발전을 나타낸다. 이러한 특수 실행 유닛은 지원되는 수치 형식(Numerical Format)을 사용하여 행렬 곱셈-누산 연산(Matrix Multiply-Accumulate Operations)을 수행한다. 텐서 코어는 밀집 행렬 연산(Dense Matrix Computation)을 가속하여 트랜스포머 어텐션(Transformer Attention), 합성곱 신경망(Convolutional Neural Network), 대규모 언어 모델(Large Language Model)의 성능을 향상시킨다. 실제 효과는 데이터 유형, 행렬 크기 및 커널 구현 방식에 따라 달라진다.

GPU의 연산 성능은 메모리 계층 구조(Memory Hierarchy)에 크게 의존한다. 전역 메모리(Global Memory)는 큰 저장 용량을 제공하지만 일반적으로 온칩 메모리보다 접근 지연시간이 길다. 레지스터(Register)는 스레드 전용 고속 저장 공간을 제공하며, 공유 메모리(Shared Memory)는 동일 블록 내부 스레드 사이의 효율적인 통신과 데이터 재사용을 지원한다. L2 캐시(L2 Cache)는 반복적인 외부 메모리 접근을 줄여 GPU 작업 부하의 메모리 접근 효율을 향상시킨다.

메모리 대역폭(Memory Bandwidth)은 산술 연산 처리량만큼 중요한 성능 요소이다. GPU에 강력한 실행 유닛이 있더라도 데이터를 충분히 빠르게 공급하지 못하면 연산 자원이 제대로 활용되지 않는다. 메모리 접근 병합(Memory Coalescing), 캐시 지역성(Cache Locality), 공유 메모리 타일링(Shared-Memory Tiling), 레지스터 재사용(Register Reuse)은 불필요한 데이터 전송을 줄인다. 따라서 효율적인 GPU 프로그래밍에는 연산량, 메모리 트래픽(Memory Traffic), 동기화 오버헤드(Synchronization Overhead), 병렬성(Parallelism)의 균형이 필요하다.

산술 집약도(Arithmetic Intensity)와 메모리 대역폭의 관계는 작업 부하가 연산 병목(Compute-Bound)인지 메모리 병목(Memory-Bound)인지를 결정한다. 산술 집약도는 특정 메모리 계층에서 전송된 바이트(Byte)당 수행되는 연산 수를 의미한다. 데이터 재사용이 제한된 작업은 대역폭 제약을 자주 경험하지만, 최적화된 행렬 연산은 충분한 병렬성과 적절한 수치 정밀도(Numerical Precision)가 확보되면 GPU의 이론적 연산 성능에 근접할 수 있다.

GPU 아키텍처는 실행 효율(Execution Efficiency), 수치 연산 기능(Numerical Capabilities), 메모리 시스템(Memory System), 스케줄링(Scheduling)을 개선하는 세대별 하드웨어 발전을 통해 진화해 왔다. NVIDIA 볼타(Volta)는 텐서 코어(Tensor Core)와 독립적 스레드 스케줄링(Independent Thread Scheduling)을 도입하여 현대 인공지능 가속의 중요한 기반을 마련하였다. 이후 아키텍처는 이러한 기능을 확장하면서 점점 더 복잡하고 대규모화되는 연산 작업을 처리하도록 발전하였다.

NVIDIA 암페어(Ampere)는 텐서플로트-32(TensorFloat-32, TF32)와 지원 장치에서의 구조적 희소성 가속(Structured Sparsity Acceleration)을 포함하여 텐서 코어 기능을 확장하였다. 이러한 기능은 기존 프로그래밍 인터페이스(Programming Interface)를 유지하면서 다양한 딥러닝(Deep Learning) 연산의 효율을 향상시켰다. 암페어는 메모리와 실행 자원도 개선하였지만, 구체적인 기능은 데이터센터용 GPU와 소비자용 GPU 구현에 따라 차이가 있다.

에이다(Ada) 아키텍처는 아키텍처 개선과 첨단 제조 공정(Advanced Manufacturing Process)을 통해 그래픽 처리, 인공지능 추론(AI Inference), 에너지 효율(Energy Efficiency)을 향상시키는 데 중점을 둔다. 텐서 코어는 가속된 기계학습(Machine Learning) 연산을 지원하며, 여러 구현에서 확대된 캐시(Cache)는 외부 메모리 트래픽을 감소시킨다. 에이다 기반 GPU는 아키텍처 개선이 대화형 애플리케이션(Interactive Application)과 인공지능 작업 모두에 이점을 제공할 수 있음을 보여준다.

NVIDIA 호퍼(Hopper)는 대규모 인공지능 학습과 고성능 컴퓨팅(High-Performance Computing, HPC)을 위한 GPU 가속 기능을 확장한다. 호퍼는 트랜스포머 엔진(Transformer Engine), FP8 연산, 비동기 데이터 이동(Asynchronous Data Movement), 스레드 블록 협력(Thread-Block Cooperation)을 위한 고급 메커니즘을 도입하였다. 이러한 기능은 소프트웨어 프레임워크와 커널이 하드웨어 기능을 효과적으로 활용할 때 복잡한 텐서 작업의 연산 자원 활용률을 향상시키도록 설계되었다.

GPU 최적화(GPU Optimization)에는 실행 자원, 메모리 접근, 수치 연산에 대한 통합적인 관리가 필요하다. 스레드 블록 크기(Thread-Block Dimensions), 워프 점유율(Warp Occupancy), 레지스터 할당(Register Allocation), 공유 메모리 사용량(Shared-Memory Consumption)은 동시에 실행할 수 있는 작업의 수에 영향을 미친다. 점유율을 높이는 것이 항상 성능 향상으로 이어지지는 않으며, 과도한 병렬성은 근본적인 병목을 해결하지 못한 채 자원 압박(Resource Pressure)이나 스케줄링 오버헤드를 증가시킬 수 있다.

커널 최적화(Kernel Optimization)는 개별 GPU 연산의 실행시간을 줄이는 데 초점을 맞춘다. 주요 기법에는 연산 융합(Operation Fusion), 루프 펼치기(Loop Unrolling), 벡터화된 메모리 접근(Vectorized Memory Access), 효율적인 타일링(Tiling), 동기화 최소화가 포함된다. 융합 커널(Fused Kernel)은 여러 연산을 하나의 실행 과정으로 결합하여 중간 메모리 전송을 제거할 수 있다. 그러나 융합은 레지스터 압박, 명령어 복잡성, 스케줄링 유연성 감소와 균형을 이루어야 한다.

수치 정밀도(Numerical Precision)는 GPU 성능, 메모리 요구량(Memory Requirements), 계산 정확도(Computational Accuracy)에 직접적인 영향을 미친다. FP32는 다양한 과학 및 기계학습 연산에서 중요하며, FP16, BF16, FP8은 지원되는 텐서 연산을 가속하고 메모리 트래픽을 줄일 수 있다. 혼합 정밀도(Mixed Precision) 기법은 여러 수치 형식을 결합하여 적절한 누산(Accumulation)과 수치 안정성(Numerical Stability)을 유지하면서 연산 효율을 향상시킨다.

파이토치(PyTorch)와 텐서플로(TensorFlow) 같은 딥러닝 프레임워크(Deep Learning Framework)는 GPU 라이브러리와 런타임 시스템(Runtime System)을 사용하여 계산 그래프(Computational Graph)를 효율적으로 실행한다. cuBLAS와 cuDNN은 일반적인 수학 연산에 대해 최적화된 구현을 제공한다. 프레임워크 컴파일러(Framework Compiler)와 실행 엔진(Execution Engine)은 커널 선택, 그래프 최적화, 연산 융합, 하드웨어 인식 스케줄링(Hardware-Aware Scheduling)을 통해 성능을 추가로 향상시킨다.

GPU 아키텍처는 대규모 신경망 학습에서 핵심적인 역할을 수행한다. 순전파(Forward Propagation), 기울기 계산(Gradient Computation), 매개변수 갱신(Parameter Update)은 병렬 실행의 이점을 활용하는 대규모 행렬 및 텐서 연산을 포함한다. 학습 성능은 GPU의 산술 연산 능력뿐 아니라 메모리 용량, 배치 크기(Batch Size), 옵티마이저 상태(Optimizer State), 통신 오버헤드(Communication Overhead), 실행 유닛에 지속적으로 유효한 작업을 공급하는 능력에도 의존한다.

추론(Inference) 단계에서 GPU는 서로 다른 성능 제약조건(Performance Constraints)에 따라 모델 실행을 가속한다. 대화형 애플리케이션은 응답 지연시간(Response Latency)을 중시하지만, 배치 처리(Batch Processing)는 처리량(Throughput)과 하드웨어 활용률(Hardware Utilization)을 강조한다. 양자화(Quantization), 배칭(Batching), 커널 융합(Kernel Fusion), 메모리 효율적 어텐션(Memory-Efficient Attention)은 추론 효율을 향상시킨다. 최적 구성은 모델 아키텍처, 시퀀스 길이(Sequence Length), 동시 요청 수(Concurrency), 배포 요구사항(Deployment Requirements)에 따라 달라진다.

다중 GPU 시스템(Multi-GPU System)은 개별 가속기의 한계를 넘어 연산 용량을 확장한다. 데이터 병렬화(Data Parallelism)는 학습 샘플을 여러 장치에 분산하며, 텐서 병렬화(Tensor Parallelism)와 파이프라인 병렬화(Pipeline Parallelism)는 모델 연산을 분할한다. PCIe와 NVLink 같은 통신 기술은 가속기 사이의 데이터 교환을 지원한다. 효율적인 확장에는 연산, 동기화, 인터커넥트 대역폭(Interconnect Bandwidth), 메모리 용량, 통신 지연시간(Communication Latency)의 균형이 필요하다.

과학 시뮬레이션(Scientific Simulation)도 많은 물리 모델이 넓은 공간 영역에서 반복적인 수치 계산을 수행하므로 GPU 병렬 처리의 이점을 활용한다. 유체역학(Fluid Dynamics), 입자 시뮬레이션(Particle Simulation), 유한요소법(Finite-Element Method), 수치 최적화(Numerical Optimization)는 GPU 실행 자원을 효과적으로 사용할 수 있다. 성능은 연산의 규칙성(Computational Regularity), 메모리 지역성(Memory Locality), 정밀도 요구사항, 수치 알고리즘의 대규모 병렬 처리 적합성에 따라 결정된다.

로보틱스(Robotics) 분야에서는 인식(Perception), 위치 추정(Localization), 지도 작성(Mapping), 시뮬레이션(Simulation), 학습 기반 제어(Learning-Based Control)에 GPU 활용이 증가하고 있다. 영상 처리(Image Processing), 포인트 클라우드 분석(Point-Cloud Analysis), 신경망 추론(Neural Inference), 물리 시뮬레이션(Physics Simulation)은 GPU에서 효율적으로 실행될 수 있다. 그러나 로봇 시스템은 결정론적 타이밍(Deterministic Timing), 전력 소비, 열 제약조건(Thermal Constraints), 실시간 동작을 담당하는 CPU 및 임베디드 제어기(Embedded Controller)와의 협력도 고려해야 한다.

실제 GPU 성능은 이론적인 초당 부동소수점 연산 수(Floating-Point Operations Per Second, FLOPS)만으로 평가할 수 없다. 메모리 대역폭, 실행 점유율(Execution Occupancy), 커널 실행 오버헤드(Kernel Launch Overhead), 캐시 동작(Cache Behavior), 데이터 전송 비용(Data Transfer Cost)이 애플리케이션 성능에 큰 영향을 미친다. NVIDIA 엔사이트 시스템즈(Nsight Systems)와 엔사이트 컴퓨트(Nsight Compute) 같은 프로파일링 도구(Profiling Tool)는 실행 병목을 식별하고 최적화 대상이 산술 연산, 메모리 이동, 스케줄링 중 어디에 있는지 분석하도록 지원한다.

전력 소비(Power Consumption)와 열 특성(Thermal Behavior) 역시 중요한 아키텍처 고려사항이다. 높은 연산 활용률은 전력 및 냉각 요구량을 증가시키며, 지속적인 작업 부하는 전력 제한(Power Limit)이나 온도 제한(Temperature Limit)에 따른 동작 주파수(Frequency) 조정을 유발할 수 있다. 따라서 와트당 성능(Performance-per-Watt)은 에너지 공급과 냉각 용량이 실제 운용 조건을 제한하는 데이터센터(Data Center), 자율 시스템(Autonomous System), 임베디드 인공지능 플랫폼(Embedded AI Platform)에서 중요한 평가 지표가 된다.

GPU 아키텍처는 궁극적으로 병렬 실행(Parallel Execution), 특수 산술 연산 유닛(Specialized Arithmetic Unit), 메모리 구성(Memory Organization), 프로그래밍 추상화(Programming Abstraction), 시스템 수준 통신(System-Level Communication) 사이의 통합적인 관계로 이해해야 한다. 효율적인 인공지능 연산은 알고리즘과 소프트웨어가 이러한 하드웨어 특성을 일관되게 활용할 때 실현된다. 이러한 기반은 GPU 실행 모델, 텐서 코어, 메모리 최적화, 아키텍처 세대별 발전, 실제 인공지능 응용에 대한 심층적인 이해로 이어진다.

## 01.01 GPU vs CPU

![](images/image2.png){width="7.268055555555556in" height="7.268055555555556in"}

중앙 처리 장치(Central Processing Unit, CPU)와 그래픽 처리 장치(Graphics Processing Unit, GPU)는 서로 보완적인 두 가지 컴퓨팅 아키텍처(Computing Architecture)를 나타낸다. CPU는 개별 작업의 지연시간(Latency)을 최소화하면서 다양한 명령어를 효율적으로 실행하도록 설계되었다. 반면 GPU는 다수의 유사한 연산을 병렬 실행 자원(Parallel Execution Resources)에 분산하여 전체 처리량(Aggregate Throughput)을 극대화한다. 이러한 차이는 인공지능 작업 부하(AI Workload)의 구현과 최적화 방식에 영향을 미친다.

CPU는 일반적으로 순차 실행(Sequential Execution)과 복잡한 제어 흐름(Control Flow)에 최적화된 비교적 적은 수의 고성능 처리 코어(Processing Core)를 포함한다. 각 코어에는 고급 명령어 스케줄링(Instruction Scheduling), 분기 예측(Branch Prediction), 캐싱(Caching), 비순차 실행(Out-of-Order Execution) 메커니즘이 포함된다. 이러한 기능을 통해 CPU는 예측하기 어려운 작업 부하, 운영체제(Operating System) 활동, 데이터베이스 트랜잭션(Database Transaction), 빈번한 의사결정이 필요한 애플리케이션을 효율적으로 처리한다.

GPU는 다수의 병렬 산술 실행 유닛(Parallel Arithmetic Execution Unit)을 더 큰 처리 구조(Processing Structure)로 구성한다. GPU는 개별 명령어 흐름(Instruction Stream)의 성능을 극대화하기보다 수많은 스레드(Thread)를 동시에 실행하여 높은 전체 연산 처리량(Computational Throughput)을 달성한다. 이러한 구조는 대규모 배열(Array), 행렬(Matrix), 텐서(Tensor), 독립적인 데이터 요소에 대해 반복적인 연산을 수행하는 작업에서 특히 효과적이다.

CPU와 GPU 설계의 근본적인 차이는 지연시간 최적화(Latency Optimization)와 처리량 최적화(Throughput Optimization)에 있다. CPU는 개별 명령어 시퀀스(Instruction Sequence)를 빠르게 완료하는 데 집중하지만, GPU는 일정 시간 동안 완료되는 전체 연산량을 강조한다. GPU는 사용 가능한 스레드 그룹(Thread Group)을 전환하여 개별 메모리 접근 지연시간(Memory Access Latency)을 허용하면서 데이터 전송이 진행되는 동안 다른 연산을 실행할 수 있다.

CPU 아키텍처는 높은 명령어 수준 병렬성(Instruction-Level Parallelism)을 유지하기 위해 정교한 제어 메커니즘(Control Mechanism)에 크게 의존한다. 분기 예측(Branch Prediction)은 프로그램의 실행 경로를 예측하고, 추측 실행(Speculative Execution)은 가능성이 높은 명령어 경로를 처리하며, 비순차 실행(Out-of-Order Execution)은 의존성이 허용하는 범위에서 독립적인 연산의 실행 순서를 조정한다. 이러한 기술은 응답성을 향상시키지만 상당한 하드웨어 자원, 전력 소비, 아키텍처 복잡성을 요구한다.

GPU 아키텍처는 스레드 수준 병렬성(Thread-Level Parallelism)과 데이터 수준 병렬성(Data-Level Parallelism)을 강조한다. 대규모 스레드 집합은 서로 다른 데이터 요소에 대해 유사한 명령어를 실행하여 다수의 산술 연산 유닛(Arithmetic Unit)을 효율적으로 활용한다. NVIDIA GPU는 일반적으로 32개의 스레드로 구성된 워프(Warp)를 사용한다. 단일 명령어 다중 스레드(Single Instruction, Multiple Threads, SIMT) 실행 모델은 명령어 실행을 조정하면서 각 스레드가 독립적인 레지스터(Register)와 실행 상태(Execution State)를 유지하도록 한다.

제어 흐름 분기(Control-Flow Divergence)는 CPU와 GPU의 성능에 서로 다른 영향을 미친다. CPU는 전용 예측 및 스케줄링 메커니즘을 통해 복잡한 분기를 처리하지만, 분기 예측 실패(Branch Misprediction)가 발생하면 성능 저하가 발생한다. GPU는 동일한 워프 내부의 스레드가 유사한 실행 경로를 따를 때 가장 효율적이다. 스레드 사이에 분기가 발생하면 서로 다른 분기 경로를 별도로 실행해야 할 수 있으며, 활성 스레드의 비율이 감소하여 유효 처리량(Effective Throughput)이 낮아진다.

메모리 아키텍처(Memory Architecture) 역시 CPU와 GPU를 구분하는 중요한 요소이다. CPU는 일반적으로 불규칙한 메모리 접근 패턴(Irregular Memory Access Pattern)의 지연시간을 줄이기 위해 정교한 캐시 계층 구조(Cache Hierarchy)를 사용한다. GPU는 높은 전체 메모리 대역폭(Aggregate Memory Bandwidth)과 다수의 동시 실행 스레드에 대한 효율적인 접근을 강조한다. GPU 메모리 시스템은 장치 메모리(Device Memory), 공유 메모리(Shared Memory), 레지스터, 캐시(Cache)를 결합하여 병렬 실행과 반복적인 데이터 재사용을 지원한다.

현대 CPU는 AVX와 같은 벡터 명령어(Vector Instruction) 및 기타 단일 명령어 다중 데이터(Single Instruction, Multiple Data, SIMD) 확장 기능을 지원하여 여러 데이터 요소를 동시에 처리한다. GPU는 이러한 병렬 처리 방식을 훨씬 더 많은 실행 유닛과 스레드로 확장한다. 따라서 CPU는 중간 규모의 수치 연산(Numerical Workload)을 효율적으로 수행할 수 있지만, 충분한 병렬 작업이 존재하면 GPU가 더 높은 가속 성능을 제공하는 경우가 많다.

산술 집약도(Arithmetic Intensity)는 어떤 프로세서 아키텍처가 더 적합한지를 결정하는 중요한 요소이다. 전송된 바이트(Byte)당 많은 수학 연산을 수행하는 작업은 GPU의 높은 연산 능력을 효과적으로 활용할 수 있다. 그러나 작은 데이터셋(Dataset), 빈번한 동기화(Synchronization), 불규칙한 메모리 접근, 제한된 병렬성을 포함하는 애플리케이션은 GPU 실행 및 데이터 이동에 추가 오버헤드(Overhead)가 발생하므로 CPU에서 더 높은 성능을 보일 수 있다.

행렬 곱셈(Matrix Multiplication)은 GPU의 아키텍처적 장점을 보여주는 대표적인 사례이다. 딥러닝 모델(Deep Learning Model)은 순전파(Forward Propagation)와 기울기 계산(Gradient Computation) 과정에서 대규모 행렬 곱셈을 반복적으로 수행한다. GPU는 이러한 연산을 병렬 실행 유닛에 분산하고, 지원되는 수치 형식(Numerical Format)에 대해 특수한 텐서 코어(Tensor Core)를 사용할 수 있다. CPU도 동일한 수학 연산을 수행할 수 있지만, 대규모 밀집 행렬 연산(Dense Matrix Workload)은 적절한 GPU에서 더 높은 처리량을 달성하는 경우가 많다.

CPU 메모리와 GPU 메모리 사이의 데이터 전송(Data Transfer)은 애플리케이션 성능에 상당한 영향을 미칠 수 있다. 개별 GPU(Discrete GPU)를 사용하는 시스템에서 PCIe를 통한 텐서(Tensor) 전송은 지연시간을 발생시키고 통신 대역폭(Communication Bandwidth)을 소비한다. 작은 연산 작업에서는 이러한 오버헤드가 GPU 가속의 이점을 상쇄할 수 있다. 따라서 효율적인 애플리케이션은 불필요한 전송을 최소화하고 재사용 가능한 데이터를 GPU에 유지하며 가능한 경우 여러 연산을 결합한다.

커널 실행 오버헤드(Kernel Launch Overhead)는 두 아키텍처를 구분하는 또 다른 요소이다. CPU 함수(Function)는 일반적인 프로그램 제어 흐름에서 직접 실행할 수 있지만, GPU 커널(Kernel)은 런타임 스케줄링(Runtime Scheduling)과 장치 실행 관리(Device Execution Management)가 필요하다. 커널 실행 오버헤드는 대규모 연산에 비해 작을 수 있지만, 매우 짧은 커널을 반복적으로 실행하면 효율이 감소할 수 있다. 커널 융합(Kernel Fusion)과 비동기 실행(Asynchronous Execution)은 이러한 한계를 완화하는 데 도움이 된다.

CPU는 일반적으로 이기종 컴퓨팅 시스템(Heterogeneous Computing System)에서 제어 프로세서(Control Processor) 역할을 수행한다. CPU는 운영체제 서비스, 애플리케이션 로직(Application Logic), 메모리 할당(Memory Allocation), 입출력(Input/Output, I/O) 작업, GPU 작업 스케줄링을 관리한다. GPU는 높은 병렬성이 필요한 작업을 위한 연산 가속기(Computational Accelerator)로 동작한다. 이러한 역할 분담을 통해 애플리케이션은 유연한 순차 실행과 특화된 고처리량 연산을 결합할 수 있다.

딥러닝 학습(Deep Learning Training)은 CPU와 GPU의 협력이 중요한 이유를 보여준다. CPU는 일반적으로 데이터셋 로딩(Dataset Loading), 전처리(Preprocessing), 학습 제어(Training Control), 저장 시스템(Storage System)과의 통신을 조정한다. GPU는 순전파, 역전파(Backpropagation), 기울기 계산을 포함하는 연산 집약적인 텐서 연산(Compute-Intensive Tensor Operations)을 수행한다. 효율적인 학습 파이프라인(Training Pipeline)은 데이터 준비 및 전송과 GPU 실행을 중첩하여 유휴 시간(Idle Time)을 줄인다.

추론 작업 부하(Inference Workload)는 학습과 다른 아키텍처적 절충관계(Architectural Tradeoff)를 가진다. GPU는 다수의 텐서 연산을 동시에 처리할 수 있으므로 대규모 모델, 높은 동시 요청 수(Request Concurrency), 큰 배치 크기(Batch Size)에 효과적이다. 반면 CPU는 소형 모델(Compact Model), 적은 요청량(Low Request Volume), 엄격한 자원 제약(Resource Constraints)이 있는 애플리케이션에 더 적합할 수 있다. 실제 성능은 모델 구조, 수치 정밀도(Numerical Precision), 메모리 용량, 실행 소프트웨어에 따라 달라진다.

전력 효율(Power Efficiency)은 프로세서 종류만으로 판단하기보다 작업 부하의 특성을 기준으로 평가해야 한다. GPU는 대규모 병렬 연산에서 단위 에너지당 더 많은 연산을 완료하여 높은 에너지 효율(Energy Efficiency)을 제공할 수 있다. 반면 GPU 자원을 충분히 활용하지 못하는 작거나 불규칙한 작업에서는 CPU가 더 적은 총에너지를 소비할 수 있다. 시스템 수준 측정(System-Level Measurement)에는 메모리, 통신, 냉각, 유휴 전력(Idle Power)을 포함해야 한다.

확장성(Scalability) 역시 CPU와 GPU 시스템에서 서로 다르게 나타난다. 멀티코어 CPU(Multi-Core CPU)는 범용 코어(General-Purpose Core)를 추가하고 공유 메모리 조정(Shared-Memory Coordination)을 활용하여 실행 용량을 확장한다. 다중 GPU 시스템(Multi-GPU System)은 PCIe와 NVLink 같은 기술을 사용하여 여러 가속기에 연산을 분산한다. GPU 확장은 상당한 처리량 향상을 제공할 수 있지만, 통신 오버헤드, 동기화, 작업 부하 분할(Workload Partitioning)이 확장 효율을 제한할 수 있다.

프로그래밍 모델(Programming Model)은 이러한 아키텍처 차이를 반영한다. CPU 애플리케이션은 일반적으로 기존 스레드(Thread), 작업 스케줄러(Task Scheduler), 벡터화 라이브러리(Vectorized Library)를 사용한다. GPU 애플리케이션은 CUDA, HIP 또는 가속기 지향 컴파일러 프레임워크(Accelerator-Oriented Compiler Framework)와 같은 전문 프로그래밍 환경을 사용한다. 효율적인 GPU 프로그래밍에는 스레드 구성, 메모리 접근 병합(Memory Coalescing), 동기화, 레지스터 압박(Register Pressure), 실행 자원 사이의 작업 분산을 고려해야 한다.

성능 비교(Performance Comparison)를 위해서는 동등한 조건에서 신중하게 수행한 벤치마킹(Benchmarking)이 필요하다. CPU와 GPU의 결과는 수치 정밀도, 배치 크기, 메모리 전송, 전처리, 전체 실행시간을 고려해야 한다. 최대 부동소수점 연산 성능(Peak Floating-Point Performance)만으로 실제 우수성을 판단할 수는 없다. 종단 간 지연시간(End-to-End Latency), 처리량, 에너지 소비, 하드웨어 활용률(Hardware Utilization), 애플리케이션 정확도(Application Accuracy)는 더욱 의미 있는 비교 기준을 제공한다.

과학 컴퓨팅(Scientific Computing)에서 CPU는 불규칙한 알고리즘(Irregular Algorithm), 복잡한 의사결정 구조(Decision Structure), 광범위한 순차 의존성(Sequential Dependency)을 포함하는 작업에 여전히 중요하다. GPU는 밀집 선형대수(Dense Linear Algebra), 구조화된 수치 시뮬레이션(Structured Numerical Simulation), 영상 처리(Image Processing), 대규모 병렬 연산에 강점을 가진다. 하이브리드 구현(Hybrid Implementation)은 불규칙한 제어 연산을 CPU에 할당하고 연산 집약적인 수치 커널(Numerical Kernel)을 GPU에서 가속하는 경우가 많다.

로보틱스(Robotics)와 자율 시스템(Autonomous System)은 두 프로세서의 상호 보완적인 관계를 잘 보여준다. CPU는 예측 가능한 실행 동작(Predictable Execution Behavior)이 필요한 시스템 조정(System Coordination), 통신, 경로 계획(Planning), 제어(Control) 작업을 처리한다. GPU는 인식 모델(Perception Model), 영상 분석(Image Analysis), 포인트 클라우드 처리(Point-Cloud Processing), 신경망 추론(Neural Network Inference)을 가속한다. 신뢰성 높은 로봇 시스템은 연산 요구사항, 시간 제약조건(Timing Constraints), 사용 가능한 하드웨어 자원에 따라 작업을 분배한다.

인공지능 모델의 복잡성이 증가하면서 이기종 컴퓨팅(Heterogeneous Computing)의 중요성도 더욱 커지고 있다. CPU는 범용적인 유연성(General-Purpose Flexibility)을 제공하고, GPU는 높은 처리량의 병렬 연산(High-Throughput Parallel Computation)과 특수한 텐서 가속(Tensor Acceleration)을 제공한다. 두 아키텍처의 절충관계를 이해하면 엔지니어는 적절한 하드웨어를 선택하고 데이터 이동을 최소화하며 실행 파이프라인(Execution Pipeline)을 최적화할 수 있다. 이를 통해 성능, 에너지 효율, 비용, 신뢰성(Reliability)의 균형을 갖춘 확장 가능한 인공지능 시스템을 설계할 수 있다.

## 01.02 Streaming Multiprocessor (SM)

![](images/image3.png){width="7.268055555555556in" height="7.268055555555556in"}

스트리밍 멀티프로세서(Streaming Multiprocessor, SM)는 NVIDIA GPU 아키텍처의 기본 실행 단위로, 대량의 병렬 스레드(Parallel Thread)를 처리하는 역할을 담당한다. 각 SM은 산술 실행 유닛(Arithmetic Execution Unit), 워프 스케줄러(Warp Scheduler), 레지스터(Register), 공유 메모리(Shared Memory), 명령어 발행 메커니즘(Instruction Dispatch Mechanism)을 통합한다. 이러한 자원을 조정하여 다수의 독립적인 연산을 동시에 실행함으로써 복잡한 인공지능 작업 부하(AI Workload)에서 높은 연산 처리량(Computational Throughput)을 달성한다.

일반적인 CPU 코어(CPU Core)와 달리 SM은 단일 명령어 흐름(Instruction Stream)을 최적화하기보다 수많은 경량 스레드(Lightweight Thread)를 관리하도록 설계되었다. 하드웨어 자원이 허용하면 여러 스레드 블록(Thread Block)이 하나의 SM에 상주할 수 있다. 스레드는 워프(Warp) 단위로 구성되며, NVIDIA 아키텍처에서는 일반적으로 32개의 스레드가 하나의 워프를 형성한다. 이러한 구조는 실행 작업을 효율적으로 분산하면서 명령어 및 메모리 지연시간(Latency)을 허용할 수 있는 충분한 병렬 작업을 유지하도록 한다.

SM의 내부 구조(Internal Structure)는 GPU 세대에 따라 달라지지만, 기본 구성요소는 일관된 목적을 수행한다. 산술 유닛(Arithmetic Unit)은 수치 연산(Numerical Operation)을 수행하고, 로드/스토어 유닛(Load/Store Unit)은 메모리 명령어(Memory Instruction)를 관리하며, 특수 실행 자원(Specialized Execution Resource)은 지원되는 수학 함수를 가속한다. 워프 스케줄러는 실행 가능한 명령어를 선택하고, 레지스터와 공유 메모리는 활성 스레드 및 협력하는 스레드 블록에 저지연 저장 공간(Low-Latency Storage)을 제공한다.

워프 스케줄링(Warp Scheduling)은 GPU 작업에서 메모리 및 명령어 의존성(Instruction Dependency)이 자주 발생하기 때문에 SM 성능의 핵심 요소이다. 특정 워프가 실행을 계속할 수 없으면 스케줄러는 다른 준비된 워프(Ready Warp)의 명령어를 발행할 수 있다. 이러한 지연시간 은닉(Latency Hiding) 메커니즘은 모든 메모리 연산이 즉시 완료되지 않더라도 유효한 연산을 계속 수행하도록 한다. 효과는 사용 가능한 병렬성(Parallelism), 자원 점유율(Resource Occupancy), 실행 가능한 워프의 수에 따라 달라진다.

SM은 레지스터, 프로그램 카운터(Program Counter), 스케줄링 정보(Scheduling Information)를 포함하여 여러 상주 워프(Resident Warp)의 실행 상태(Execution State)를 유지한다. 하드웨어 자원은 동시에 활성화할 수 있는 워프와 스레드 블록의 수를 결정한다. 최대 상주 스레드 수(Maximum Resident Threads)는 GPU 아키텍처와 구체적인 구현에 따라 달라진다. 따라서 최적의 커널 구성(Kernel Configuration)은 모든 GPU 세대가 동일한 SM 용량을 가진다고 가정하기보다 장치별 제한사항(Device-Specific Limits)을 고려해야 한다.

레지스터(Register)는 중간 값(Intermediate Value), 루프 변수(Loop Variable), 임시 연산 결과(Temporary Computational Result)를 저장하는 고속 스레드 전용 저장 공간(Thread-Private Storage)을 제공한다. 각 SM은 상주 스레드가 공유하는 제한된 레지스터 파일(Register File)을 포함한다. 스레드당 많은 레지스터를 요구하는 커널은 동시에 활성화할 수 있는 워프의 수를 감소시킬 수 있다. 과도한 레지스터 요구량은 로컬 메모리(Local Memory)로의 스필링(Spilling)을 유발하여 메모리 트래픽(Memory Traffic)을 증가시키고 실행 효율을 저하시킬 수 있다.

공유 메모리(Shared Memory)는 동일한 블록 내부의 스레드가 데이터를 효율적으로 교환하도록 지원하는 명시적 관리 방식의 온칩 저장 자원(On-Chip Storage Resource)이다. 공유 메모리는 타일 기반 행렬 곱셈(Tiled Matrix Multiplication), 합성곱(Convolution), 리덕션(Reduction), 반복적인 데이터 재사용이 필요한 알고리즘에서 특히 유용하다. 자주 접근하는 데이터를 공유 메모리에 적재하면 전역 메모리 트랜잭션(Global Memory Transaction)을 줄일 수 있다. 다른 스레드가 기록한 데이터에 의존하는 경우 올바른 동기화(Synchronization)가 필요하다.

많은 NVIDIA 아키텍처는 공유 메모리와 L1 캐시(L1 Cache) 사이의 구성 가능한 관계를 제공하지만, 구체적인 구조는 세대별로 다르다. L1 캐싱(L1 Caching)은 특정 메모리 접근을 가속하고, 공유 메모리는 프로그래머가 직접 제어하는 데이터 배치(Data Placement)를 지원한다. 개별 스레드 블록에 더 많은 공유 메모리를 할당하면 데이터 지역성(Data Locality)이 향상될 수 있지만, 하나의 SM에서 동시에 실행할 수 있는 블록 수는 감소할 수 있다.

SM의 점유율(Occupancy)은 하드웨어가 지원하는 최대 활성 워프 수에 대해 현재 상주하는 워프의 비율을 나타낸다. 점유율은 스레드 블록 크기(Thread-Block Size), 레지스터 사용량(Register Usage), 공유 메모리 할당량(Shared-Memory Allocation), 아키텍처 자원 제한(Architectural Resource Limits)에 따라 결정된다. 높은 점유율은 실행 가능한 워프를 더 많이 제공하여 지연시간 은닉을 향상시킬 수 있다. 그러나 실행 처리량이나 메모리 대역폭이 이미 포화된 경우 최대 점유율이 반드시 최대 성능을 의미하지는 않는다.

스레드 블록(Thread Block)은 GPU 스케줄링 하드웨어(GPU Scheduling Hardware)에 의해 SM에 할당되며, 해당 블록의 스레드는 할당된 멀티프로세서(Multiprocessor)의 자원을 사용하여 실행된다. 일반적으로 블록은 실행이 완료될 때까지 할당된 SM과 연결된 상태를 유지한다. 하나의 SM에서 여러 블록이 동시에 실행될 수 있으며, 추가 블록은 자원이 사용 가능해질 때까지 대기한다. 이러한 실행 모델은 프로그래머가 개별 SM을 직접 선택하지 않고도 확장 가능한 커널(Scalable Kernel)을 구현하도록 지원한다.

스레드 블록과 SM 자원 사이의 관계는 커널 효율(Kernel Efficiency)에 직접적인 영향을 미친다. 큰 블록은 상당한 협력 연산(Cooperative Computation)과 데이터 재사용(Data Reuse)을 제공할 수 있지만, 더 많은 레지스터와 공유 메모리를 소비한다. 작은 블록은 더 높은 스케줄링 유연성(Scheduling Flexibility)을 제공할 수 있으나 추가 오버헤드를 발생시키거나 협력 기회를 줄일 수 있다. 적절한 블록 크기를 선택하려면 병렬성, 자원 소비(Resource Consumption), 동기화, 작업 부하 특성(Workload Characteristics)의 균형을 고려해야 한다.

명령어 발행 메커니즘(Instruction Dispatch Mechanism)은 워프 스케줄링 결정을 SM의 실행 파이프라인(Execution Pipeline)에 연결한다. 서로 다른 명령어는 부동소수점 연산(Floating-Point Arithmetic), 정수 연산(Integer Computation), 메모리 연산(Memory Operation), 특수 수학 연산 파이프라인(Specialized Mathematical Pipeline) 등 서로 다른 기능 유닛(Functional Unit)을 사용할 수 있다. 하드웨어 세대마다 명령어 발행 능력(Issue Capability)과 실행 자원이 다르다. 효율적인 활용을 위해서는 충분한 독립 명령어와 적절한 연산 조합이 필요하다.

부동소수점 실행 유닛(Floating-Point Execution Unit)은 GPU 아키텍처에서 정의한 정밀도 형식(Precision Format)에 따라 수치 연산을 지원한다. FP32 연산은 그래픽, 과학 컴퓨팅(Scientific Computing), 기계학습(Machine Learning)에서 널리 사용되며, 지원되는 FP64 기능은 GPU 제품에 따라 상당한 차이가 있다. 정수 실행 유닛(Integer Execution Unit)은 주소 계산(Address Calculation), 인덱싱(Indexing), 논리 연산(Logical Operation), 정수 산술 연산(Integer Arithmetic)을 처리한다. 이러한 자원 사이의 구성 비율은 작업 부하별 성능에 영향을 미친다.

텐서 코어(Tensor Core)는 지원되는 NVIDIA GPU 세대의 SM에 통합된 특수 연산 유닛(Specialized Computational Unit)이다. 텐서 코어는 지원되는 데이터 형식(Data Format)과 피연산자 구성(Operand Configuration)을 사용하여 행렬 곱셈-누산 연산(Matrix Multiply-Accumulate Operation)을 가속한다. 딥러닝 프레임워크(Deep Learning Framework)는 행렬 곱셈, 합성곱, 트랜스포머 연산(Transformer Computation)에 이러한 유닛을 활용한다. 높은 텐서 코어 활용률(Tensor Core Utilization)을 달성하려면 적절한 행렬 형태(Matrix Shape), 메모리 배치(Memory Layout), 정밀도 선택, 최적화된 커널이 필요하다.

로드/스토어 유닛(Load/Store Unit)은 레지스터와 GPU 메모리 계층 구조(Memory Hierarchy) 사이에서 데이터를 전송하는 명령어를 관리한다. 분산되거나 중복된 메모리 트랜잭션(Memory Transaction)은 유효 대역폭(Effective Bandwidth)을 감소시킬 수 있으므로 효율적인 메모리 접근 패턴(Memory Access Pattern)이 중요하다. 병합된 전역 메모리 접근(Coalesced Global Memory Access)은 인접 스레드의 적절한 요청을 효율적인 메모리 트랜잭션으로 결합한다. 공유 메모리 접근에서도 뱅크 충돌(Bank Conflict)로 인한 불필요한 직렬화(Serialization)를 방지하기 위해 뱅크 구성(Bank Organization)을 고려해야 한다.

동기화 메커니즘(Synchronization Mechanism)은 협력하는 스레드를 조정하고 공유 데이터에 의존하는 연산의 정확성을 보장한다. 블록 수준 배리어(Block-Level Barrier)는 참여 스레드가 필요한 실행 지점에 도달한 후 의존 연산(Dependent Operation)이 진행되도록 한다. 메모리 순서 규칙(Memory Ordering Rule)은 기록된 데이터가 다른 스레드에 언제 가시화되는지를 결정한다. 과도한 동기화는 병렬 실행 효율을 감소시키지만, 불충분한 동기화는 경쟁 상태(Race Condition)와 잘못된 연산 결과를 발생시킬 수 있다.

제어 흐름 분기(Control-Flow Divergence)는 동일한 워프 내부의 스레드가 서로 다른 실행 경로(Execution Path)를 따를 때 발생한다. SM은 실행 모델(Execution Model)에 따라 이러한 경로를 관리해야 하며, 특정 시점에 유효한 작업을 수행하는 스레드 수가 감소할 수 있다. 분기 경로에 상당한 연산량이 포함되어 있으면 분기 비용이 특히 커진다. 유사한 제어 동작(Control Behavior)을 수행하는 스레드를 함께 구성하는 알고리즘은 실행 효율을 향상시키고 낭비되는 명령어 처리 용량(Instruction Capacity)을 줄일 수 있다.

SM 성능은 개별 스레드 내부의 명령어 수준 병렬성(Instruction-Level Parallelism)에도 의존한다. 독립적인 산술 연산(Independent Arithmetic Operation)은 의존성이 허용하는 경우 실행을 중첩하여 효율적으로 처리할 수 있다. 긴 의존성 체인(Dependency Chain)은 산술 유닛이 사용 가능한 상태에서도 명령어 발행(Instruction Issue)을 제한할 수 있다. 명령어 수준 병렬성과 스레드 수준 병렬성(Thread-Level Parallelism)을 결합하면 반복적인 수학 연산을 포함하는 계산 커널(Computational Kernel)의 실행 처리량을 유지하는 데 도움이 된다.

현대 SM 아키텍처는 연산과 데이터 이동(Data Movement)을 조정하기 위한 점점 더 정교한 메커니즘을 지원한다. GPU 세대에 따라 비동기 메모리 연산(Asynchronous Memory Operation)은 데이터 전송과 산술 연산 실행을 중첩할 수 있다. 하드웨어 지원 동기화(Hardware-Supported Synchronization)와 특수 전송 메커니즘(Specialized Transfer Mechanism)은 명령어 오버헤드를 줄일 수 있다. 이러한 기능은 전역 메모리와 온칩 저장 공간 사이에서 데이터를 반복적으로 이동시키는 타일 기반 텐서 작업(Tiled Tensor Workload)에 특히 유용하다.

SM의 확장성(SM Scalability)은 하나의 GPU 장치에 다수의 멀티프로세서를 통합하여 연산 용량을 증가시킬 수 있도록 한다. 대규모 커널은 스레드 블록을 사용 가능한 SM에 분산하여 상당한 병렬 실행을 수행한다. 그러나 전체 성능이 SM 수에 비례하여 선형적으로 증가하는 것은 아니다. 전역 메모리 대역폭(Global Memory Bandwidth), 캐시 용량(Cache Capacity), 동기화 요구사항(Synchronization Requirements), 작업 부하 크기(Workload Size)는 추가 연산 자원이 존재하더라도 활용률을 제한할 수 있다.

볼타(Volta), 암페어(Ampere), 에이다(Ada), 호퍼(Hopper)와 같은 GPU 세대는 SM 구성과 기능이 지속적으로 발전하고 있음을 보여준다. 주요 개선사항에는 특수 텐서 연산(Specialized Tensor Computation), 스케줄링 개선(Scheduling Refinement), 메모리 시스템 향상(Memory-System Enhancement), 실행과 데이터 이동을 중첩하기 위한 추가 메커니즘이 포함된다. SM 구성은 제품별로 다르므로 성능 튜닝(Performance Tuning)은 아키텍처 계열(Architectural Family)의 이름에만 의존하지 않고 실제 장치 사양(Device Specification)을 기준으로 수행해야 한다.

프로파일링(Profiling)은 SM 활용률(SM Utilization)과 실행 병목(Execution Bottleneck)에 관한 필수 정보를 제공한다. NVIDIA 엔사이트 컴퓨트(NVIDIA Nsight Compute)는 달성 점유율(Achieved Occupancy), 명령어 처리량(Instruction Throughput), 메모리 동작(Memory Behavior), 레지스터 사용량(Register Usage), 실행 파이프라인 활동(Execution Pipeline Activity)과 관련된 지표를 보고할 수 있다. 이러한 측정값은 병렬성 부족, 메모리 지연(Memory Stall), 의존성 제한(Dependency Limitation), 자원 포화(Resource Saturation)를 구분하는 데 도움이 된다. 최적화 결정은 점유율만이 아니라 관측된 병목을 기준으로 이루어져야 한다.

딥러닝 애플리케이션(Deep Learning Application)에서 SM은 신경망 연산(Neural Network Operation)의 기반이 되는 계산 커널을 실행한다. 행렬 곱셈(Matrix Multiplication), 합성곱(Convolution), 정규화(Normalization), 활성화 함수(Activation Function), 기울기 계산(Gradient Calculation)은 다수의 멀티프로세서 사이에서 조정된 실행에 의존한다. 커널 효율은 이러한 연산이 레지스터, 공유 메모리, 산술 파이프라인(Arithmetic Pipeline), 텐서 코어를 얼마나 효과적으로 활용하는지에 따라 결정된다. 따라서 높은 학습 및 추론 처리량을 달성하려면 효율적인 스케줄링과 데이터 재사용이 필수적이다.

SM 아키텍처에 대한 이해는 효율적인 GPU 프로그래밍(GPU Programming)과 성능 엔지니어링(Performance Engineering)의 기반을 제공한다. 워프 스케줄링, 레지스터 할당(Register Allocation), 공유 메모리 구성(Shared-Memory Organization), 실행 파이프라인, 점유율은 상호작용하여 연산 효율(Computational Efficiency)을 결정한다. 최적화된 커널은 이러한 자원의 균형을 유지하면서 불필요한 데이터 이동과 동기화를 최소화한다. 이러한 아키텍처 이해는 확장 가능한 인공지능 학습(Scalable AI Training), 효율적인 추론(Efficient Inference), 과학 컴퓨팅(Scientific Computing), 고성능 로봇 인식 시스템(High-Performance Robotic Perception System)의 구현을 지원한다.

## 01.03 Warp and Thread Model

![](images/image4.png){width="7.268055555555556in" height="7.268055555555556in"}

워프 및 스레드 모델(Warp and Thread Model)은 NVIDIA GPU 아키텍처에서 병렬 실행(Parallel Execution)의 기반을 형성한다. 개별 명령어 흐름(Instruction Stream)의 성능을 강조하는 CPU와 달리 GPU는 수천 개의 경량 스레드(Lightweight Thread)로 연산을 구성한다. 이러한 스레드는 그리드(Grid), 스레드 블록(Thread Block), 워프(Warp)로 이루어진 계층 구조(Hierarchical Structure) 내에서 실행된다. 이러한 구성을 이해하는 것은 효율적인 CUDA 커널(CUDA Kernel)을 개발하고 인공지능 작업 부하(AI Workload)를 최적화하는 데 필수적이다.

스레드(Thread)는 CUDA 프로그래밍 모델(CUDA Programming Model)에서 가장 작은 논리적 실행 단위(Logical Execution Entity)를 나타낸다. 각 스레드는 자체 레지스터(Register), 지역 변수(Local Variable), 실행 상태(Execution State)를 사용하여 커널 명령어(Kernel Instruction)를 실행한다. 스레드는 인덱싱 메커니즘(Indexing Mechanism)을 통해 할당된 데이터를 식별하므로 동일한 커널이 배열(Array)이나 텐서(Tensor)의 서로 다른 요소를 처리할 수 있다. 이러한 접근 방식은 개별 데이터 요소마다 별도의 프로그램을 정의하지 않고도 확장 가능한 병렬 연산(Scalable Parallel Computation)을 지원한다.

스레드는 협력 실행(Cooperative Execution)의 기본 단위를 제공하는 스레드 블록(Thread Block)으로 그룹화된다. 동일한 블록 내부의 스레드는 공유 메모리(Shared Memory)를 통해 데이터를 교환하고 동기화 기본 연산(Synchronization Primitive)을 사용하여 작업을 조정할 수 있다. 블록은 1차원(One-Dimensional), 2차원(Two-Dimensional), 3차원(Three-Dimensional)의 스레드 배열을 가질 수 있다. 블록당 스레드 수는 아키텍처 제한(Architectural Limit)과 커널이 요구하는 자원에 의해 제약된다.

스레드 블록은 커널 실행(Kernel Launch)의 전체 병렬 실행 영역(Parallel Execution Domain)을 정의하는 그리드(Grid)로 구성된다. 그리드 차원(Grid Dimension)은 연산에 참여하는 블록 수를 결정하고, 블록 차원(Block Dimension)은 각 블록 내부의 스레드 구성을 지정한다. CUDA 인덱싱 변수(CUDA Indexing Variable)를 사용하면 스레드는 블록 좌표와 스레드 좌표로부터 전역 위치(Global Position)를 계산할 수 있다. 이러한 계층 구조는 물리적인 GPU 멀티프로세서(Multiprocessor)의 수와 관계없이 대규모 데이터셋(Dataset)을 처리하도록 지원한다.

각 스레드 블록 내부에서 NVIDIA GPU는 일반적으로 연속된 32개의 스레드를 하나의 워프(Warp)로 구성한다. 워프는 하드웨어가 스레드 그룹에 명령어를 발행하는 기본 스케줄링 단위(Fundamental Scheduling Unit)이다. 워프 내부의 스레드는 단일 명령어 다중 스레드(Single Instruction, Multiple Threads, SIMT) 모델에 따라 실행된다. 스레드는 조정된 명령어 실행(Coordinated Instruction Execution)을 공유하지만, 각 스레드는 자체 레지스터, 조건자(Predicate), 논리적 실행 상태(Logical Execution State)를 유지한다.

단일 명령어 다중 스레드(SIMT) 실행 모델은 기존의 단일 명령어 다중 데이터(Single Instruction, Multiple Data, SIMD) 벡터 처리(Vector Processing)와 차이가 있다. SIMT는 개별 제어 흐름(Control Flow)을 가진 독립적으로 주소 지정 가능한 스레드(Independently Addressable Thread)를 제공하지만, SIMD는 일반적으로 여러 데이터 요소에 대한 벡터 연산(Vector Operation)을 표현한다. 하드웨어는 병렬 실행 효율을 높이기 위해 워프 스레드의 실행을 조정한다. 이러한 추상화(Abstraction)는 프로그래머가 스레드 수준 논리(Thread-Level Logic)를 표현하는 동안 GPU가 그룹화된 명령어 실행을 관리하도록 한다.

워프 스케줄링(Warp Scheduling)은 메모리 및 명령어 의존성(Instruction Dependency)이 존재하더라도 스트리밍 멀티프로세서(Streaming Multiprocessor, SM)가 연산 처리량(Computational Throughput)을 유지하도록 한다. 각 멀티프로세서는 아키텍처 자원 제한(Architectural Resource Limit)에 따라 여러 상주 워프(Resident Warp)를 지원한다. 하나의 워프가 실행을 진행할 수 없는 의존성을 만나면 스케줄러는 다른 실행 가능한 워프(Eligible Warp)를 선택할 수 있다. 이러한 빠른 스케줄링 메커니즘은 진행 중인 메모리 연산과 유효한 계산을 중첩하여 지연시간을 은닉(Latency Hiding)한다.

워프 실행 효율(Warp Execution Efficiency)은 참여 스레드 사이의 제어 흐름 유사성(Control-Flow Similarity)에 따라 달라진다. 모든 활성 스레드(Active Thread)가 동일한 명령어 경로(Instruction Path)를 따르면 하드웨어는 실행 자원(Execution Resource)을 효과적으로 활용할 수 있다. 분기 발산(Divergence)은 워프 내부의 스레드가 조건문(Conditional Statement)에서 서로 다른 분기 경로를 선택할 때 발생한다. GPU는 각 경로에 필요한 스레드만 참여시켜 실행해야 하므로 실행 경로가 다시 합류(Reconvergence)할 때까지 처리량이 감소할 수 있다.

분기 발산(Branch Divergence)은 불규칙한 의사결정(Irregular Decision)이나 데이터 의존 조건(Data-Dependent Condition)을 포함하는 알고리즘에서 특히 중요하다. 예를 들어 인접한 스레드가 가변 길이 시퀀스(Variable-Length Sequence)나 희소 데이터 구조(Sparse Data Structure)를 처리할 때 서로 다른 분기를 선택할 수 있다. 유사한 실행 경로를 유도하도록 데이터를 구성하면 분기 발산을 줄일 수 있다. 조건부 실행(Predication)과 분기 구조 재구성(Branch Restructuring)도 효율을 향상시킬 수 있지만, 효과는 명령어 비용과 작업 부하 특성에 따라 달라진다.

병렬 연산(Parallel Operation)이 중간 결과(Intermediate Result)를 공유하거나 조정된 실행(Coordinated Execution)에 의존하는 경우 스레드 동기화(Thread Synchronization)가 필요하다. CUDA는 참여 스레드가 계속 실행하기 전에 공통 실행 지점(Common Execution Point)에 도달하도록 하는 블록 수준 동기화(Block-Level Synchronization) 메커니즘을 제공한다. 이러한 배리어(Barrier)는 일반적으로 공유 메모리 쓰기(Shared-Memory Write) 이후에 사용된다. 올바른 동기화는 데이터 위험(Data Hazard)을 방지하지만, 과도한 배리어는 대기시간을 발생시키고 병렬 실행 기회를 감소시킨다.

워프 수준 동기화(Warp-Level Synchronization)는 단일 워프 내부 스레드 사이의 협력을 지원한다. CUDA는 값 교환(Value Exchange), 조건 투표(Voting), 참여 레인(Participating Lane)의 조정을 위한 워프 수준 기본 연산(Warp-Level Primitive)을 제공한다. 셔플 연산(Shuffle Operation)은 공유 메모리 트랜잭션(Shared-Memory Transaction) 없이 레인 사이에서 레지스터 값을 전달할 수 있다. 이러한 기능은 리덕션(Reduction), 스캔(Scan), 집합 연산(Collective Computation)에 유용하지만, 활성 스레드 마스크(Active-Thread Mask)와 동기화 요구사항을 준수해야 한다.

레인 식별자(Lane Identifier)는 워프 내부에서 스레드의 위치를 나타내며, 워프 식별자(Warp Identifier)는 블록 내부의 스레드 그룹을 구분한다. 이러한 개념은 워프 수준 알고리즘(Warp-Level Algorithm)을 설계하고 작업을 효율적으로 분배하는 데 도움이 된다. 스레드 인덱싱(Thread Indexing)은 블록 차원과 부분 워프(Partial Warp)를 고려해야 한다. 특히 블록 크기나 데이터셋 경계(Dataset Boundary)로 인해 일부 레인이 비활성화될 수 있으므로 모든 워프에 32개의 활성 스레드가 존재한다고 가정해서는 안 된다.

메모리 접근 패턴(Memory Access Pattern)은 워프 구성(Warp Organization)과 밀접하게 관련된다. 인접한 스레드가 연속된 메모리 주소(Consecutive Memory Address)에 접근하면 하드웨어는 해당 요청을 효율적인 전역 메모리 트랜잭션(Global Memory Transaction)으로 결합할 수 있다. 이러한 동작을 메모리 접근 병합(Memory Coalescing)이라고 한다. 분산된 메모리 접근(Scattered Memory Access)은 추가 트랜잭션을 발생시켜 유효 대역폭(Effective Bandwidth)을 감소시킬 수 있다. 따라서 데이터 배치(Data Layout)와 스레드 인덱싱은 GPU 메모리 최적화(Memory Optimization)의 중요한 요소이다.

공유 메모리 접근(Shared-Memory Access)도 워프 내부의 스레드가 메모리 뱅크(Memory Bank)에 접근하는 방식에 따라 영향을 받는다. 공유 메모리는 호환 가능한 접근 요청을 효율적으로 처리할 수 있는 여러 뱅크로 분할된다. 여러 스레드가 동일한 뱅크에 매핑된 서로 다른 주소를 요청하면 뱅크 충돌(Bank Conflict)이 발생하여 트랜잭션이 직렬화(Serialization)될 수 있다. 신중하게 설계된 인덱싱과 데이터 배치는 이러한 충돌을 줄이고 타일 기반 행렬 연산(Tiled Matrix Operation)과 협력 알고리즘(Cooperative Algorithm)의 성능을 향상시킨다.

스레드 수준 병렬성(Thread-Level Parallelism)은 지연시간 은닉(Latency Hiding)과 실행 처리량(Execution Throughput)에 직접적으로 기여한다. 활성 워프(Active Warp)가 부족한 커널은 메모리 연산이나 의존 명령어(Dependent Instruction)를 기다리는 동안 실행 유닛(Execution Unit)이 유휴 상태로 남을 수 있다. 사용 가능한 워프 수를 늘리면 활용률(Utilization)이 향상될 수 있지만 하드웨어 자원에는 한계가 있다. 레지스터 할당(Register Allocation), 공유 메모리 사용량(Shared-Memory Consumption), 스레드 블록 차원(Thread-Block Dimension)은 동시에 상주할 수 있는 워프 수를 결정한다.

점유율(Occupancy)은 스트리밍 멀티프로세서가 지원하는 최대 활성 워프 수에 대해 현재 상주하는 워프의 비율을 나타낸다. 높은 점유율은 추가적인 스케줄링 기회(Scheduling Opportunity)를 제공할 수 있지만 반드시 더 높은 성능을 보장하지는 않는다. 일부 커널은 충분한 명령어 수준 병렬성(Instruction-Level Parallelism)이나 효율적인 데이터 재사용(Data Reuse)을 통해 중간 수준의 점유율에서도 높은 처리량을 달성한다. 따라서 최적화는 점유율 자체를 극대화하기보다 실제 실행 병목(Execution Bottleneck)을 평가해야 한다.

스레드 블록 크기(Thread-Block Size)는 스케줄링 유연성(Scheduling Flexibility)과 자원 활용률(Resource Utilization)에 영향을 미친다. 워프 크기의 배수로 구성된 블록은 일반적으로 블록 내부에서 일부만 채워진 워프(Partially Filled Warp)를 방지한다. 그러나 지나치게 큰 블록은 많은 레지스터나 공유 메모리를 소비하여 동시 블록 상주(Concurrent Block Residency)를 제한할 수 있다. 작은 블록은 스케줄링 유연성을 향상시킬 수 있지만 협력 효율(Cooperation Efficiency)을 감소시킬 수 있다. 최적 구성은 커널 동작(Kernel Behavior)과 하드웨어 특성(Hardware Characteristics)에 따라 결정된다.

NVIDIA 볼타(Volta)에서 도입된 독립적 스레드 스케줄링(Independent Thread Scheduling)은 워프 내부의 스레드 실행 상태를 더욱 유연하게 관리하도록 실행 모델을 개선하였다. 이러한 기능은 일부 분기 실행 패턴(Divergent Execution Pattern)에 대한 지원을 향상시키지만 분기 발산의 성능 비용을 제거하지는 않는다. 스레드가 데이터를 교환하거나 메모리를 통해 협력하는 경우 프로그램은 암묵적인 동기 실행(Implicit Lockstep Behavior)에 의존하지 않고 적절한 동기화 메커니즘을 사용해야 한다.

워프 수준 프로그래밍(Warp-Level Programming)에서는 활성 마스크(Active Mask)와 실행 참여(Execution Participation)를 신중하게 처리해야 한다. 집합 연산(Collective Operation)은 어떤 레인이 참여하는지 명시해야 하며, 참여 스레드는 해당 연산의 실행 요구사항을 충족해야 한다. 스레드 활성 상태(Thread Activity)에 대한 잘못된 가정은 정의되지 않은 동작(Undefined Behavior)이나 잘못된 결과를 발생시킬 수 있다. 명시적 동기화(Explicit Synchronization)와 명확한 참여 규칙(Participation Rule)은 분기 발산이나 불규칙한 작업 부하를 포함하는 알고리즘에서 특히 중요하다.

워프 및 스레드 모델은 병렬 리덕션(Parallel Reduction)의 구현 방식에 큰 영향을 미친다. 스레드는 처음에 독립적인 데이터 요소를 처리한 다음 공유 메모리 또는 워프 수준 연산(Warp-Level Operation)을 통해 중간 결과를 결합한다. 효율적인 구현은 동기화 오버헤드(Synchronization Overhead)와 불필요한 메모리 트래픽(Memory Traffic)을 줄인다. 유사한 원리는 접두사 스캔(Prefix Scan), 히스토그램 계산(Histogram Computation), 정규화(Normalization), 기계학습 시스템에서 자주 사용되는 기타 집합 연산에도 적용된다.

딥러닝 작업 부하(Deep Learning Workload)는 신중하게 구성된 스레드 매핑(Thread Mapping)을 통해 성능상의 이점을 얻는다. 행렬 곱셈 커널(Matrix Multiplication Kernel)은 입력 및 출력 행렬의 타일(Tile)에 스레드와 워프를 할당하여 데이터 재사용과 조정된 산술 실행(Coordinated Arithmetic Execution)을 지원한다. 합성곱 커널(Convolution Kernel)도 공간 위치(Spatial Position), 채널(Channel), 출력 요소(Output Element)를 병렬 스레드에 분산한다. 효율적인 매핑은 메모리 지역성(Memory Locality), 실행 활용률(Execution Utilization), 특수 텐서 코어(Tensor Core) 연산과의 호환성을 향상시킨다.

불규칙한 작업 부하(Irregular Workload)는 워프 실행(Warp Execution)에 추가적인 어려움을 발생시킨다. 그래프 처리(Graph Processing), 희소 행렬 연산(Sparse Matrix Computation), 가변 길이 시퀀스 처리(Variable-Length Sequence Processing)는 불균등한 작업 분배(Uneven Work Distribution)와 분기 발산을 포함할 수 있다. 부하 균형 기법(Load-Balancing Technique)은 작업을 스레드와 워프에 더욱 균등하게 할당할 수 있다. 지속 실행 커널(Persistent Kernel)과 작업 큐(Work Queue)는 기존의 정적 매핑(Static Mapping)이 상당한 불균형을 발생시키는 경우 활용률을 추가로 향상시킬 수 있다.

프로파일링 도구(Profiling Tool)는 워프 실행 동작과 스레드 수준 효율(Thread-Level Efficiency)을 분석하는 데 필요한 정보를 제공한다. NVIDIA 엔사이트 컴퓨트(NVIDIA Nsight Compute)는 활성 워프(Active Warp), 실행 가능한 워프(Eligible Warp), 분기 동작(Branch Behavior), 메모리 트랜잭션(Memory Transaction), 실행 정지(Execution Stall)와 관련된 지표를 제공할 수 있다. 이러한 측정값은 병렬성 부족, 분기 발산, 비효율적인 메모리 접근, 명령어 의존성 중 어떤 요소가 성능을 제한하는지 식별하는 데 도움이 된다. 프로파일링 결과는 커널 구조 재설계(Kernel Restructuring)와 실행 구성(Launch Configuration) 선택의 기준이 되어야 한다.

워프 및 스레드 모델은 궁극적으로 CUDA의 프로그래밍 추상화(Programming Abstraction)를 GPU 하드웨어의 물리적 실행 자원(Physical Execution Resource)과 연결한다. 스레드는 독립적인 연산을 표현하고, 블록은 협력 실행을 지원하며, 워프는 효율적인 실행을 위한 스케줄링 구조(Scheduling Structure)를 제공한다. 인덱싱(Indexing), 분기 발산(Divergence), 동기화(Synchronization), 메모리 접근(Memory Access), 점유율(Occupancy)을 이해하면 개발자는 딥러닝, 과학 컴퓨팅(Scientific Computing), 로보틱스(Robotics) 전반에서 GPU 병렬성을 효과적으로 활용하는 확장 가능한 커널(Scalable Kernel)을 설계할 수 있다.

## 01.04 CUDA Programming Model [w/Code]

![](images/image5.png){width="7.268055555555556in" height="7.268055555555556in"}

CUDA 프로그래밍 모델(CUDA Programming Model)은 NVIDIA GPU에서 대규모 병렬 연산(Massively Parallel Computation)을 실행하기 위한 체계적인 접근 방식을 제공한다. CUDA는 기존 프로그래밍 방식에 장치 커널(Device Kernel), 스레드 구성(Thread Organization), 메모리 관리(Memory Management), 동기화(Synchronization)를 위한 추상화(Abstraction)를 추가한다. 개발자는 개별 하드웨어 실행 유닛(Hardware Execution Unit)을 직접 관리하지 않고도 병렬 알고리즘(Parallel Algorithm)을 구현할 수 있다. 이러한 설계는 상위 수준 소프트웨어 연산(High-Level Software Operation)을 GPU의 기본 아키텍처와 연결한다.

CUDA 애플리케이션은 일반적으로 CPU가 호스트(Host) 역할을 하고 GPU가 장치(Device)로 동작하는 이기종 컴퓨팅 모델(Heterogeneous Computing Model)을 따른다. 호스트는 애플리케이션 로직(Application Logic), 메모리 할당(Memory Allocation), 커널 실행(Kernel Launch)을 관리하고, 장치는 연산 집약적인 병렬 작업(Compute-Intensive Parallel Operation)을 수행한다. 이러한 역할 분담을 통해 애플리케이션은 순차 처리(Sequential Processing)와 높은 처리량의 가속 연산(High-Throughput Acceleration)을 결합할 수 있다. 전체 성능을 높이려면 호스트와 장치 사이의 효율적인 조정이 필수적이다.

CUDA 커널(CUDA Kernel)은 다수의 GPU 스레드(Thread)가 병렬로 실행하는 함수(Function)이다. 프로그래머는 각 스레드가 수행할 연산을 정의하고, CUDA 런타임(CUDA Runtime)은 사용 가능한 하드웨어 자원에 실행 작업을 배치한다. 커널 실행 시에는 스레드 블록(Thread Block)의 수와 각 블록에 포함되는 스레드 수를 지정한다. 이러한 구성은 개별 멀티프로세서(Multiprocessor)에 작업을 직접 할당하는 것이 아니라 GPU에 제공할 논리적 병렬성(Logical Parallelism)을 결정한다.

CUDA는 병렬 실행(Parallel Execution)을 스레드(Thread), 블록(Block), 그리드(Grid)의 계층 구조(Hierarchical Structure)로 구성한다. 스레드는 개별 연산을 수행하고, 블록은 협력하는 스레드를 그룹화하며, 그리드는 하나의 커널 실행에 참여하는 블록을 포함한다. 각 스레드는 내장 인덱싱 변수(Built-In Indexing Variable)를 통해 자신의 위치를 식별한다. 이러한 계층 구조는 스트리밍 멀티프로세서(Streaming Multiprocessor)의 수가 서로 다른 GPU에서도 실행할 수 있는 확장 가능한 알고리즘(Scalable Algorithm)을 지원한다.

스레드 인덱싱(Thread Indexing)은 논리적 실행 위치(Logical Execution Position)를 애플리케이션 데이터와 연결한다. 1차원 인덱싱(One-Dimensional Indexing)은 일반적으로 벡터 연산(Vector Operation)에 사용되며, 2차원 및 3차원 배열은 영상(Image), 행렬(Matrix), 체적 데이터셋(Volumetric Dataset)에 적합하다. 올바른 인덱싱은 스레드가 의도한 데이터 요소를 처리하면서 유효하지 않은 메모리 위치(Invalid Memory Location)에 접근하지 않도록 보장한다. 데이터셋 크기가 선택한 실행 구성(Execution Configuration)과 정확히 일치하지 않는 경우 경계 검사(Boundary Check)가 필요하다.

CUDA 실행 모델(CUDA Execution Model)은 워프(Warp)를 기본 하드웨어 스케줄링 그룹(Hardware Scheduling Group)으로 사용한다. NVIDIA GPU는 일반적으로 32개의 스레드를 하나의 워프로 구성하며, 단일 명령어 다중 스레드(Single Instruction, Multiple Threads, SIMT) 실행 방식을 따른다. 워프 내부의 스레드는 개별 실행 상태(Execution State)를 유지하면서 조정된 명령어(Coordinated Instruction)를 실행한다. 인접한 스레드가 서로 다른 제어 경로(Control Path)를 따르면 분기 발산(Branch Divergence)으로 효율이 감소할 수 있으므로 규칙적인 연산 패턴(Regular Computation Pattern)이 유리하다.

CUDA는 서로 다른 성능 특성과 가시성 규칙(Visibility Rule)을 가진 여러 메모리 공간(Memory Space)을 제공한다. 레지스터(Register)는 고속 스레드 전용 저장 공간(Thread-Private Storage)을 제공하며, 로컬 메모리(Local Memory)는 레지스터에 모두 유지할 수 없는 스레드 전용 데이터를 저장한다. 공유 메모리(Shared Memory)는 블록 내부 스레드 사이의 통신을 지원한다. 전역 메모리(Global Memory)는 여러 커널에서 접근할 수 있는 장치 전체 저장 공간(Device-Wide Storage)을 제공하며, 상수 메모리(Constant Memory)와 텍스처 메모리(Texture Memory)는 특수한 접근 패턴을 지원한다.

전역 메모리(Global Memory)는 일반적으로 큰 저장 용량을 제공하지만 온칩 저장 공간(On-Chip Storage)보다 접근 지연시간(Access Latency)이 길다. 따라서 효율적인 CUDA 커널은 인접한 스레드의 메모리 요청을 적은 수의 메모리 트랜잭션(Memory Transaction)으로 처리할 수 있는 병합 메모리 접근(Coalesced Memory Access)을 추구한다. 비효율적으로 구성된 메모리 접근은 메모리 트래픽(Memory Traffic)을 증가시키고 유효 대역폭(Effective Bandwidth)을 감소시킨다. 데이터 배치(Data Layout)와 스레드 매핑(Thread Mapping)은 지역성(Locality)을 향상시키고 중복 전송을 최소화하도록 함께 설계해야 한다.

공유 메모리(Shared Memory)는 협력하는 스레드가 명시적으로 관리하는 온칩 저장 공간(On-Chip Storage)을 제공한다. 공유 메모리는 행렬 곱셈(Matrix Multiplication), 합성곱(Convolution), 리덕션(Reduction) 연산에서 재사용 가능한 데이터를 캐싱(Caching)하는 데 자주 사용된다. 스레드는 전역 메모리에서 공유 메모리로 데이터를 적재하고 반복 연산을 수행한 다음, 다른 스레드가 생성한 결과를 사용하기 전에 동기화할 수 있다. 효율적인 공유 메모리 활용은 외부 메모리 트래픽을 줄이지만 용량 제한(Capacity Limit)과 뱅크 충돌(Bank Conflict)을 고려해야 한다.

CUDA 동기화 메커니즘(Synchronization Mechanism)은 병렬 연산이 공유 결과(Shared Result)에 의존할 때 정확성을 유지한다. 블록 수준 배리어(Block-Level Barrier)는 블록 내부 스레드를 조정하며, 워프 수준 기본 연산(Warp-Level Primitive)은 참여 레인(Participating Lane) 사이의 통신을 지원한다. 장치 전체 동기화(Device-Wide Coordination)는 일반적으로 별도의 커널 경계(Kernel Boundary)나 특수한 협력 실행 메커니즘(Cooperative Execution Mechanism)을 필요로 한다. 불필요한 대기는 병렬 효율을 감소시키고 참여 규칙(Participation Rule)을 위반하면 교착 상태(Deadlock)가 발생할 수 있으므로 동기화는 신중하게 사용해야 한다.

CUDA 런타임(CUDA Runtime)은 장치 메모리(Device Memory)의 할당, 전송, 해제를 위한 함수를 제공한다. 기존의 명시적 메모리 관리(Explicit Memory Management)는 호스트 메모리 할당(Host Allocation)과 장치 메모리 할당(Device Allocation)을 구분하며, 두 주소 공간(Address Space) 사이에서 데이터 전송을 요구한다. 통합 메모리(Unified Memory)는 지원되는 데이터가 프로세서 사이를 이동할 수 있는 관리형 주소 공간(Managed Address Space)을 제공하여 프로그래밍을 단순화한다. 그러나 성능 최적화에서는 데이터 이동 오버헤드(Migration Overhead)와 접근 지역성(Access Locality)을 여전히 고려해야 한다.

CUDA 스트림(CUDA Stream)은 GPU 실행을 위해 제출되는 순서가 있는 연산 시퀀스(Ordered Operation Sequence)를 나타낸다. 하나의 스트림 내부 연산은 정의된 순서 규칙(Ordering Rule)을 따르며, 독립적인 스트림은 하드웨어 자원과 의존성이 허용하는 경우 실행을 중첩할 수 있다. 비동기 메모리 전송(Asynchronous Memory Transfer)은 관련 요구사항이 충족되면 커널 실행(Kernel Execution)과 중첩될 수 있다. 따라서 스트림은 여러 연산 단계(Computational Stage)를 포함하는 애플리케이션에서 유휴 시간(Idle Time)을 줄이고 활용률(Utilization)을 향상시키는 데 도움이 된다.

CUDA 이벤트(CUDA Event)는 실행 진행 상태(Execution Progress)를 기록하고 장치 경과시간(Device Elapsed Time)을 측정하며 스트림 사이의 의존성(Dependency)을 조정하는 메커니즘을 제공한다. 이벤트는 CPU 측 시간 측정(CPU-Side Timing)에만 의존하지 않고 커널 성능을 벤치마킹(Benchmarking)하는 데 유용하다. 정확한 측정을 위해서는 적절한 동기화와 비동기 실행(Asynchronous Execution)의 특성을 고려해야 한다. 이벤트 기반 프로파일링(Event-Based Profiling)은 커널 실행시간을 호스트 오버헤드(Host Overhead), 메모리 전송, 스케줄링 지연(Scheduling Delay)과 구분하는 데 도움이 된다.

커널 구성(Kernel Configuration)은 실행 효율(Execution Efficiency)에 큰 영향을 미친다. 블록 차원(Block Dimension)은 워프 활용률(Warp Utilization), 스케줄링 유연성(Scheduling Flexibility), 블록당 필요한 자원에 영향을 준다. 레지스터 사용량(Register Consumption)과 공유 메모리 할당량(Shared-Memory Allocation)은 하나의 스트리밍 멀티프로세서에 동시에 상주할 수 있는 블록 수를 결정한다. 큰 블록이 항상 더 빠른 것은 아니며 최대 점유율(Maximum Occupancy)이 최적 처리량(Optimal Throughput)을 보장하지도 않는다. 효과적인 튜닝(Tuning)에는 병렬성, 자원 압박(Resource Pressure), 메모리 동작(Memory Behavior)의 균형이 필요하다.

CUDA는 여러 스레드가 동일한 메모리 위치에 동시에 접근할 때 공유 값을 안전하게 갱신하기 위한 원자적 연산(Atomic Operation)을 지원한다. 원자적 명령어(Atomic Instruction)는 지원되는 연산과 메모리 범위(Memory Scope)에 대해 정의된 읽기-수정-쓰기(Read-Modify-Write) 동작을 제공한다. 이러한 기능은 카운터(Counter), 히스토그램(Histogram), 병렬 누산(Parallel Accumulation)에 유용하다. 그러나 심한 경합(Contention)은 실행을 직렬화(Serialization)할 수 있으므로 일부 작업에서는 계층적 집계(Hierarchical Aggregation)나 대체 데이터 구조(Alternative Data Structure)가 더 적합하다.

오류 처리(Error Handling)는 신뢰성 높은 CUDA 개발의 필수 요소이다. 커널 실행과 런타임 연산(Runtime Operation)은 잘못된 구성(Invalid Configuration), 메모리 할당 문제(Allocation Problem), 불법 메모리 접근(Illegal Memory Access)으로 실패할 수 있다. 많은 연산이 비동기적으로 실행되기 때문에 오류는 이후의 동기화나 런타임 함수 호출(Runtime Call)에서 확인될 수 있다. 개발자는 반환 상태(Return Status)를 검사하고 메모리 경계(Memory Boundary)를 검증하며, 성능 결과를 해석하기 전에 진단 도구(Diagnostic Tool)를 사용하여 오류를 식별해야 한다.

CUDA 라이브러리(CUDA Library)는 일반적인 수치 연산(Numerical Operation)을 위한 최적화된 구성요소를 제공한다. cuBLAS는 밀집 선형대수(Dense Linear Algebra)를 가속하며, cuDNN은 합성곱과 정규화(Normalization) 같은 신경망 연산(Neural Network Operation)을 지원한다. 다른 라이브러리는 희소 행렬(Sparse Matrix), 신호 처리(Signal Processing), 병렬 알고리즘(Parallel Algorithm)을 처리한다. 검증된 라이브러리를 활용하면 모든 연산 요소(Computational Primitive)를 직접 구현하는 것보다 GPU 세대 전반에서 더 높은 성능과 이식성(Portability)을 얻는 경우가 많다.

CUDA 컴파일 과정(CUDA Compilation Process)은 소스 프로그램(Source Program)을 NVIDIA GPU 아키텍처에 적합한 실행 코드(Executable Code)로 변환한다. 장치 코드(Device Code)는 하드웨어별 명령어(Hardware-Specific Instruction)로 변환되기 전에 중간 명령어 형식(Intermediate Instruction Format)인 PTX로 표현될 수 있다. 컴파일 옵션(Compilation Option)은 최적화, 지원 아키텍처(Supported Architecture), 수치 연산 동작(Numerical Behavior)에 영향을 미친다. 이러한 과정을 이해하면 개발자는 호환성(Compatibility)을 관리하고 컴파일러 변환(Compiler Transformation)의 효과를 평가할 수 있다.

성능 최적화(Performance Optimization)는 커널이 연산량(Computation), 메모리 대역폭(Memory Bandwidth), 지연시간(Latency), 병렬성 부족(Insufficient Parallelism) 중 어떤 요소에 의해 제한되는지 식별하는 것에서 시작한다. 메모리 접근 병합(Memory Coalescing), 루프 펼치기(Loop Unrolling), 타일링(Tiling), 커널 융합(Kernel Fusion)은 서로 다른 병목(Bottleneck)을 해결한다. 데이터 재사용(Data Reuse)을 통해 산술 집약도(Arithmetic Intensity)를 높이면 효율이 향상될 수 있다. 최적화 결정은 이론적인 하드웨어 성능에 대한 가정보다 프로파일링(Profiling) 결과를 근거로 이루어져야 한다.

CUDA 그래프(CUDA Graph)는 GPU 연산 시퀀스를 재사용 가능한 실행 그래프(Reusable Execution Graph)로 표현하여 반복적인 호스트 측 작업 제출 오버헤드(Host-Side Submission Overhead)를 줄인다. 이러한 접근 방식은 애플리케이션이 유사한 연산 흐름(Computational Workflow)을 반복적으로 실행할 때 특히 유용하다. 그래프 실행(Graph Execution)은 스케줄링 효율(Scheduling Efficiency)을 향상시키고 실행 요청 비용(Launch-Related Cost)을 감소시킬 수 있다. 그러나 정확성을 유지하려면 그래프 생성(Graph Construction), 의존성, 메모리 수명(Memory Lifetime)을 신중하게 관리해야 한다.

혼합 정밀도 연산(Mixed-Precision Computation)은 CUDA 기반 인공지능 작업 부하(AI Workload)에서 널리 사용된다. 지원되는 GPU 아키텍처는 텐서 코어(Tensor Core)와 같은 특수 하드웨어를 통해 FP16, BF16, TF32, FP8 연산을 가속한다. 이러한 수치 형식(Numerical Format)은 처리량(Throughput)을 높이고 메모리 요구량(Memory Requirement)을 줄일 수 있지만, 수치 안정성(Numerical Stability)을 고려해야 한다. 적절한 누산 정밀도(Accumulation Precision)와 스케일링 전략(Scaling Strategy)은 허용 가능한 모델 정확도(Model Accuracy)를 유지하는 데 도움이 된다.

딥러닝 프레임워크(Deep Learning Framework)는 최적화된 커널, 라이브러리, 실행 런타임(Execution Runtime)을 통해 CUDA를 통합한다. 파이토치(PyTorch)와 텐서플로(TensorFlow)는 메모리와 연산 의존성(Computational Dependency)을 관리하면서 GPU에서 텐서 연산(Tensor Operation)을 스케줄링한다. 프레임워크 컴파일러(Framework Compiler)는 연산을 결합하고 특수 커널(Specialized Kernel)을 선택하며 실행 그래프(Execution Graph)를 최적화할 수 있다. CUDA의 기본 원리를 이해하면 프레임워크 추상화(Framework Abstraction) 뒤에 숨겨진 성능 제한을 분석하고 프로파일링 결과를 해석하는 데 도움이 된다.

로보틱스(Robotics)와 실시간 인공지능 시스템(Real-Time AI System)은 인식(Perception), 영상 처리(Image Processing), 포인트 클라우드 분석(Point-Cloud Analysis), 신경망 추론(Neural Inference)을 가속하기 위해 CUDA를 사용한다. GPU 연산은 복잡한 센서 작업 부하(Sensor Workload)의 처리시간을 줄일 수 있지만, 비동기 실행과 가변적인 스케줄링 지연(Variable Scheduling Delay)은 신중한 시스템 통합(System Integration)을 요구한다. CPU 제어기(CPU Controller)는 시간 제약이 엄격한 제어 기능(Timing-Critical Control Function)을 계속 담당하는 경우가 많다. 신뢰성 높은 배포(Deployment)를 위해서는 가속 성능과 지연시간 예측 가능성(Latency Predictability), 전력 소비(Power Consumption), 열 제약조건(Thermal Constraints)의 균형이 필요하다.

CUDA 프로그래밍 모델은 궁극적으로 소프트웨어에서 정의한 병렬성(Software-Defined Parallelism)을 GPU 실행 자원(GPU Execution Resource)과 연결한다. 효과적인 활용은 적절한 스레드 구성(Thread Organization), 메모리 접근 패턴(Memory Access Pattern), 동기화(Synchronization), 비동기 실행(Asynchronous Execution), 커널 구성(Kernel Configuration)에 달려 있다. 이러한 원리를 최적화된 라이브러리와 프로파일링 도구(Profiling Tool)에 결합하면 개발자는 성능과 자원 효율(Resource Efficiency)을 향상시키면서 딥러닝(Deep Learning), 과학 컴퓨팅(Scientific Computing), 시뮬레이션(Simulation), 자율 시스템(Autonomous System)을 위한 확장 가능한 애플리케이션(Scalable Application)을 구축할 수 있다.

## 01.05 Tensor Cores

![](images/image6.png){width="7.268055555555556in" height="7.268055555555556in"}

텐서 코어(Tensor Core)는 인공지능(Artificial Intelligence)에 사용되는 행렬 연산(Matrix Operation)을 가속하기 위해 NVIDIA GPU의 스트리밍 멀티프로세서(Streaming Multiprocessor, SM)에 통합된 특수 처리 장치(Specialized Processing Unit)이다. 텐서 코어는 지원되는 수치 형식(Numerical Format)을 사용하여 행렬 곱셈-누산(Matrix Multiply-Accumulate) 연산을 실행하며, 적합한 작업 부하(Workload)에서 기존 스칼라 산술 연산(Scalar Arithmetic)보다 훨씬 높은 처리량(Throughput)을 제공한다. 텐서 코어의 도입은 밀집 텐서 연산(Dense Tensor Computation)을 하드웨어 최적화의 핵심 대상으로 설정함으로써 GPU 아키텍처를 변화시켰다.

텐서 코어가 수행하는 기본 연산은 행렬 곱셈(Matrix Multiplication)과 누산(Accumulation)의 결합이다. 개념적으로 출력 행렬(Output Matrix)은 두 입력 행렬(Input Matrix)을 곱한 결과를 누산 행렬(Accumulator Matrix)에 더하여 생성된다. 이러한 연산은 신경망 계층(Neural Network Layer), 어텐션 메커니즘(Attention Mechanism), 과학 수치 알고리즘(Scientific Numerical Algorithm)에서 반복적으로 나타난다. 전용 행렬 실행 하드웨어(Dedicated Matrix Execution Hardware)는 일반 산술 유닛(Arithmetic Unit)을 통해 동일한 연산을 수행할 때 발생하는 명령어 오버헤드(Instruction Overhead)를 줄인다.

NVIDIA는 볼타 아키텍처(Volta Architecture)에서 텐서 코어를 처음 도입하였으며, 초기에는 딥러닝(Deep Learning)을 위한 혼합 정밀도 행렬 연산(Mixed-Precision Matrix Computation)에 중점을 두었다. 초기 구현은 FP16 입력 연산을 지원하면서 유용한 수치 정확도(Numerical Accuracy)를 유지하도록 설계된 누산 기능(Accumulation Capability)을 제공하였다. 이러한 아키텍처는 수치 정밀도(Numerical Precision)와 연산 처리량(Computational Throughput) 사이의 새로운 관계를 확립하였다. 이후 GPU 세대에서는 지원되는 수치 형식, 실행 기능, 소프트웨어 프레임워크(Software Framework)와의 통합이 확대되었다.

텐서 코어 실행(Tensor Core Execution)은 개별 스칼라 값(Scalar Value)이 아니라 행렬 조각(Matrix Fragment)을 대상으로 연산한다는 점에서 기존 CUDA 코어(CUDA Core) 처리와 다르다. 스레드 그룹(Thread Group)은 지원되는 프로그래밍 인터페이스(Programming Interface)를 통해 피연산자(Operand)를 제공하고 결과를 사용하도록 협력한다. 하드웨어는 여러 곱셈-누산 연산을 조정된 행렬 명령어(Coordinated Matrix Instruction)로 수행한다. 이러한 구성은 산술 연산 밀도(Arithmetic Density)를 높이고 적합한 행렬 작업 부하에서 명령어 스케줄링 부담(Instruction Scheduling Burden)을 줄인다.

혼합 정밀도 연산(Mixed-Precision Computation)은 텐서 코어 가속(Tensor Core Acceleration)의 핵심이다. 낮은 정밀도의 수치 형식(Lower-Precision Format)은 피연산자 저장 공간(Operand Storage Requirement)을 줄이고 특수 하드웨어가 지원하는 연산량을 증가시킨다. FP16과 BF16은 딥러닝에서 널리 사용되며, FP32는 일부 계산과 누산에서 여전히 중요하다. 혼합 정밀도 알고리즘(Mixed-Precision Algorithm)은 연산 효율(Computational Efficiency)과 반올림 오차(Rounding Error), 수치 표현 범위(Numerical Range), 모델 수렴 요구사항(Model Convergence Requirement) 사이의 균형을 유지한다.

FP16은 메모리 사용량(Memory Consumption)을 줄이고 행렬 처리량(Matrix Processing Throughput)을 향상시킬 수 있는 압축된 부동소수점 표현(Compact Floating-Point Representation)을 사용한다. 그러나 제한된 지수 범위(Exponent Range)로 인해 학습 과정에서 오버플로(Overflow)나 언더플로(Underflow)가 발생할 수 있다. 손실 스케일링(Loss Scaling)은 FP16을 사용할 때 일부 기울기 언더플로(Gradient Underflow) 문제를 완화할 수 있다. 효율적인 구현에서는 일부 리덕션(Reduction)과 옵티마이저 갱신(Optimizer Update)을 포함하여 수치적으로 민감한 연산(Numerically Sensitive Computation)에 적절한 고정밀도 연산(Higher-Precision Operation)을 유지한다.

BF16은 FP16보다 넓은 지수 범위를 제공하면서 압축된 저장 형식(Compact Storage Representation)을 유지한다. BF16은 FP32와 유사한 범위의 값을 표현할 수 있지만 가수 정밀도(Mantissa Precision)는 낮기 때문에 신경망 학습(Neural Network Training)에 적합하다. 지원되는 텐서 코어는 BF16 행렬 연산을 효율적으로 가속한다. FP16과 BF16 사이의 선택은 하드웨어 지원(Hardware Support), 수치적 동작(Numerical Behavior), 모델의 안정성 요구사항(Stability Requirement)에 따라 달라진다.

텐서플로트-32(TensorFloat-32, TF32)는 NVIDIA 암페어 데이터센터 GPU(Ampere Data-Center GPU)에서 텐서 코어를 사용하여 일부 FP32 행렬 연산을 가속하기 위해 도입되었다. TF32는 FP32와 관련된 지수 범위를 유지하면서 곱셈에 사용되는 입력 유효숫자 정밀도(Input Significand Precision)를 낮춘다. 누산에는 일반적으로 FP32 정밀도(Precision)가 사용된다. 이러한 접근 방식은 호환 가능한 작업 부하에서 처리량을 향상시킬 수 있지만, 더욱 엄격한 수치 정확도가 필요한 애플리케이션은 다른 정밀도 설정(Precision Setting)을 사용해야 할 수 있다.

FP8 연산(FP8 Computation)은 NVIDIA 호퍼(Hopper) 및 이후 설계를 포함하여 지원되는 텐서 코어 아키텍처에서 저정밀도 가속(Reduced-Precision Acceleration)을 확장한다. FP8 형식은 압축된 피연산자(Compact Operand)를 제공하며, 적합한 텐서 작업 부하에서 메모리 트래픽(Memory Traffic)을 크게 줄일 수 있다. FP8 형식은 수치 범위와 정밀도가 서로 다르므로 실제 구현에서는 스케일링 전략(Scaling Strategy)과 민감한 연산의 신중한 처리가 필요하다. FP8의 성능 이점은 아키텍처, 소프트웨어 지원, 작업 부하 특성에 따라 달라진다.

텐서 코어 성능(Tensor Core Performance)은 행렬 차원(Matrix Dimension), 피연산자 정렬(Operand Alignment), 메모리 배치(Memory Layout), 지원되는 명령어 형태(Instruction Shape)의 영향을 받는다. 하드웨어 실행 타일(Hardware Execution Tile)에 효율적으로 매핑되는 행렬은 일반적으로 더 높은 활용률(Utilization)을 달성한다. 불규칙한 차원(Irregular Dimension)은 패딩(Padding)이나 대체 커널 구성(Alternative Kernel Configuration)을 필요로 할 수 있으며, 이로 인해 추가 연산이 발생할 수 있다. 따라서 개발자는 텐서 연산을 최적화할 때 수학적 작업 부하 구조(Mathematical Workload Structure)와 하드웨어 실행 요구사항(Hardware Execution Requirement)을 모두 고려해야 한다.

워프 수준 협력(Warp-Level Cooperation)은 텐서 코어 프로그래밍(Tensor Core Programming)에서 중요한 역할을 한다. CUDA는 워프 수준 행렬 명령어(Warp-Level Matrix Instruction)와 상위 수준 추상화(Higher-Level Abstraction)를 포함한 지원 프로그래밍 인터페이스를 통해 행렬 연산을 제공한다. 스레드는 공동으로 행렬 조각(Matrix Fragment)을 관리하고 조정된 실행(Coordinated Execution)에 참여한다. 스레드, 피연산자, 하드웨어 자원 사이의 정확한 매핑은 GPU 아키텍처와 명령어 유형(Instruction Type)에 따라 달라지므로 이식성이 높은 고성능 구현(Portable High-Performance Implementation)은 쉽지 않다.

최신 텐서 코어 아키텍처(Modern Tensor Core Architecture)는 비동기 실행(Asynchronous Execution)과 특수 행렬 명령어(Specialized Matrix Instruction)를 점차 확대하여 지원한다. NVIDIA 호퍼(Hopper)는 워프 그룹 행렬 연산(Warp-Group Matrix Operation)과 데이터 이동(Data Movement)을 연산과 조정하는 메커니즘을 도입하였다. 이러한 기능을 활용하면 최적화된 커널(Optimized Kernel)이 피연산자 준비(Operand Preparation)와 행렬 실행(Matrix Execution)을 중첩할 수 있다. 그러나 올바른 결과와 높은 활용률을 얻으려면 동기화(Synchronization), 공유 메모리 구성(Shared-Memory Organization), 하드웨어별 제약조건(Hardware-Specific Constraint)을 신중하게 처리해야 한다.

텐서 코어가 높은 산술 처리량(Arithmetic Throughput)을 제공하더라도 메모리 대역폭(Memory Bandwidth)은 여전히 중요한 제약조건이다. 행렬 연산은 입력 피연산자가 실행 유닛(Execution Unit)에 효율적으로 전달되어야 하며, 비효율적인 데이터 전송은 특수 하드웨어의 활용률을 낮출 수 있다. 공유 메모리 타일링(Shared-Memory Tiling), 레지스터 재사용(Register Reuse), 병합 전역 메모리 접근(Coalesced Global Memory Access)은 중복 데이터 이동(Redundant Data Movement)을 줄인다. 고성능 커널은 연산량과 피연산자 적재 및 결과 저장 비용 사이의 균형을 유지한다.

산술 집약도(Arithmetic Intensity)는 텐서 코어 작업 부하가 사용 가능한 연산 자원(Computational Resource)을 얼마나 효과적으로 활용하는지 결정한다. 대규모 행렬 곱셈(Large Matrix Multiplication)은 여러 연산에서 입력값을 재사용할 수 있으므로 높은 산술 집약도를 달성한다. 반면 작은 행렬이나 데이터 재사용이 제한된 연산은 메모리 트래픽 또는 커널 실행 오버헤드(Kernel Launch Overhead)에 의해 성능이 제한될 수 있다. 따라서 이론적인 텐서 코어 처리량(Theoretical Tensor Core Throughput)을 임의의 작업 부하에서 보장되는 애플리케이션 성능(Application Performance)으로 해석해서는 안 된다.

텐서 코어는 선형 투영(Linear Projection)과 어텐션 계산(Attention Computation)에 광범위한 행렬 곱셈이 포함되는 트랜스포머 아키텍처(Transformer Architecture)에서 특히 중요하다. 쿼리(Query), 키(Key), 값(Value) 변환은 특수 행렬 하드웨어를 활용할 수 있으며, 피드포워드 네트워크(Feed-Forward Network) 내부의 밀집 연산(Dense Operation)도 마찬가지이다. 어텐션 성능은 메모리 이동, 시퀀스 길이(Sequence Length), 중간 텐서 저장(Intermediate Tensor Storage)의 영향도 받는다. 최적화된 어텐션 커널(Optimized Attention Kernel)은 텐서 코어 실행과 효율적인 메모리 관리를 결합한다.

합성곱 신경망(Convolutional Neural Network, CNN)도 텐서 코어 가속의 이점을 얻는다. 합성곱 연산(Convolution Operation)은 특수 하드웨어에 효율적으로 매핑되는 행렬 연산으로 변환하거나 구성할 수 있다. cuDNN과 같은 라이브러리는 텐서 형태(Tensor Shape), 데이터 형식(Data Format), 사용 가능한 GPU 자원에 따라 알고리즘을 선택한다. 성능은 채널 차원(Channel Dimension), 공간 구조(Spatial Structure), 메모리 배치, 합성곱 데이터를 변환하거나 구성하는 과정에서 발생하는 오버헤드의 영향을 받는다.

신경망 학습(Neural Network Training) 과정에서 텐서 코어는 반복적인 행렬 연산을 통해 순전파(Forward Propagation)와 역전파(Backward Propagation)를 가속한다. 혼합 정밀도 학습(Mixed-Precision Training)은 메모리 사용량을 줄이고 처리량을 증가시켜 사용 가능한 자원 내에서 더 큰 모델이나 배치 크기(Batch Size)를 지원할 수 있다. 그러나 학습 안정성(Training Stability)을 유지하려면 기울기 스케일링(Gradient Scaling), 누산 정밀도(Accumulation Precision), 민감한 수학 연산(Sensitive Mathematical Operation)을 고려해야 한다. 성능 향상은 수렴 동작(Convergence Behavior)과 최종 모델 정확도(Final Model Accuracy)를 함께 평가해야 한다.

추론 작업 부하(Inference Workload)는 텐서 코어를 활용하여 지원되는 모델 연산의 처리량을 높이고 응답 지연시간(Response Latency)을 줄인다. GPU 아키텍처에 따라 FP16, BF16, FP8 및 일부 정수 형식(Integer Format)이 연산을 가속할 수 있다. 양자화 추론(Quantized Inference)은 정확도를 유지하기 위해 보정(Calibration)이나 다른 기법이 필요할 수 있다. 최적의 정밀도 형식은 모델 민감도(Model Sensitivity), 배치 크기, 지연시간 목표(Latency Target), 배포 하드웨어(Deployment Hardware)의 수치 연산 기능(Numerical Capability)에 따라 결정된다.

구조적 희소성(Structured Sparsity)은 지원되는 NVIDIA 아키텍처에서 텐서 코어 성능을 추가로 향상시킬 수 있다. 일부 하드웨어 세대는 지원되는 4개 중 2개 구조(Two-Out-of-Four Structure)와 같은 특정 희소 패턴(Sparsity Pattern)을 따르는 피연산자의 행렬 연산을 가속한다. 희소 연산 가속(Sparse Acceleration)을 위해서는 호환 가능한 데이터 표현(Data Representation)과 적합한 커널이 필요하다. 희소성은 산술 연산량을 줄일 수 있지만 실제 성능 향상은 가지치기 기법(Pruning Method), 정확도 유지(Accuracy Retention), 구조적 표현을 유지하는 데 필요한 오버헤드에 따라 달라진다.

CUDA 라이브러리와 딥러닝 프레임워크(Deep Learning Framework)는 텐서 코어 기능에 편리하게 접근할 수 있는 수단을 제공한다. cuBLAS와 cuDNN은 최적화된 행렬 및 합성곱 연산을 구현하며, 파이토치(PyTorch)와 텐서플로(TensorFlow)는 지원되는 가속 실행 경로(Acceleration Path)를 통합한다. 컴파일러 기술(Compiler Technology)은 행렬 명령어를 선택하고 메모리 배치를 최적화하며 주변 연산을 융합(Fusion)할 수 있다. 그러나 실제 텐서 코어 사용 여부는 프레임워크 설정(Framework Setting), 텐서 차원(Tensor Dimension), 정밀도 정책(Precision Policy)에 크게 영향을 받는다.

텐서 코어 활용률(Tensor Core Utilization)은 이론적인 사양만으로 판단하지 않고 프로파일링(Profiling)을 통해 평가해야 한다. NVIDIA 엔사이트 컴퓨트(NVIDIA Nsight Compute)는 행렬 명령어 활동(Matrix Instruction Activity), 메모리 병목(Memory Bottleneck), 점유율 제한(Occupancy Limitation), 실행 정지(Execution Stall)를 분석할 수 있다. 낮은 활용률은 작업 부하 크기 부족, 부적절한 차원, 비효율적인 데이터 전송, 실행 자원 경쟁(Competing Execution Demand)으로 인해 발생할 수 있다. 프로파일링은 산술 스케줄링(Arithmetic Scheduling), 메모리 구성(Memory Organization), 커널 선택(Kernel Selection) 중 어떤 부분을 최적화해야 하는지 식별하는 데 도움이 된다.

NVIDIA GPU의 세대에 따라 텐서 코어 기능(Tensor Core Capability)과 성능 특성(Performance Characteristic)은 서로 다르다. 볼타(Volta)는 특수 행렬 가속(Specialized Matrix Acceleration)의 기반을 확립하였고, 암페어(Ampere)는 수치 형식 지원(Numerical Format Support)을 확대하였으며, 에이다(Ada)는 인공지능 실행 기능(AI Execution Capability)을 개선하였다. 호퍼(Hopper)는 추가적인 저정밀도 연산(Low-Precision Operation)과 비동기 행렬 기능(Asynchronous Matrix Feature)을 도입하였다. 구체적인 기능은 제품군(Product Family)과 구성(Configuration)에 따라 다르므로 소프트웨어는 지원 기능을 확인하고 대상 장치(Target Device)에 적합한 실행 전략(Execution Strategy)을 선택해야 한다.

텐서 코어는 알고리즘이 정확도 요구사항(Accuracy Requirement)을 위반하지 않으면서 가속된 행렬 연산을 활용할 수 있는 경우 과학 컴퓨팅(Scientific Computing), 시뮬레이션(Simulation), 로보틱스(Robotics)에도 적용될 수 있다. 신경망 기반 인식(Neural Perception), 영상 처리(Image Processing), 상태 추정(State Estimation), 학습 기반 제어(Learning-Based Control)는 특수 텐서 연산(Specialized Tensor Computation)의 이점을 얻을 수 있다. 그러나 수치 민감도(Numerical Sensitivity), 결정적 실행시간(Deterministic Timing), 전력 제약조건(Power Constraint)을 고려해야 한다. 모든 알고리즘이 저정밀도 연산이나 밀집 행렬 가속(Dense Matrix Acceleration)에서 동일한 이점을 얻는 것은 아니다.

텐서 코어는 현대 인공지능 시스템(Modern Artificial Intelligence System)에서 하드웨어-소프트웨어 공동 설계(Hardware-Software Co-Design)의 중요성을 보여준다. 텐서 코어의 효과는 특수 산술 유닛(Specialized Arithmetic Unit), 수치 형식(Numerical Format), 메모리 계층 구조(Memory Hierarchy), 커널 구성(Kernel Organization), 최적화된 소프트웨어 라이브러리(Optimized Software Library)의 상호작용에서 나타난다. 이러한 관계를 이해하면 엔지니어는 적절한 수치 정확도와 신뢰성 높은 시스템 동작(Reliable System Behavior)을 유지하면서 학습 처리량(Training Throughput), 추론 효율(Inference Efficiency), 연산 확장성(Computational Scalability)을 향상시킬 수 있다.

## 01.06 Memory Hierarchy

![](images/image7.png){width="7.268055555555556in" height="7.268055555555556in"}

GPU 메모리 계층 구조(GPU Memory Hierarchy)는 저장 용량(Storage Capacity), 접근 지연시간(Access Latency), 대역폭(Bandwidth), 병렬 실행 요구사항(Parallel Execution Requirement) 사이의 균형을 유지하도록 설계되었다. NVIDIA GPU는 레지스터(Register), 공유 메모리(Shared Memory), L2 캐시(L2 Cache), 전역 메모리(Global Memory)를 포함하는 여러 계층으로 메모리를 구성한다. 각 계층은 서로 다른 성능 특성(Performance Characteristic)과 접근 규칙(Access Rule)을 제공한다. 이러한 계층 사이의 상호작용을 이해하는 것은 CUDA 커널(CUDA Kernel)을 최적화하고 인공지능 작업 부하(Artificial Intelligence Workload)에서 데이터 이동(Data Movement)을 줄이는 데 필수적이다.

전역 메모리(Global Memory)는 GPU 커널이 접근할 수 있는 주요 대용량 저장 공간(Large-Capacity Storage Space)을 제공한다. 전역 메모리는 GPU 아키텍처에 따라 일반적으로 GDDR 또는 HBM과 같은 외부 장치 메모리(External Device Memory)를 사용하여 구현된다. 전역 메모리에는 모델 매개변수(Model Parameter), 입력 텐서(Input Tensor), 중간 활성화 값(Intermediate Activation), 출력 결과(Output Result)가 저장된다. 전역 메모리는 높은 대역폭을 제공하지만 온칩 메모리 자원(On-Chip Memory Resource)에 비해 접근 지연시간이 상대적으로 길다.

전역 메모리 성능(Global Memory Performance)은 병렬 스레드(Parallel Thread)가 생성하는 접근 패턴(Access Pattern)에 크게 의존한다. 인접한 스레드가 연속되거나 적절하게 정렬된 메모리 주소(Aligned Memory Address)에 접근하면 메모리 접근 병합(Memory Coalescing)을 통해 요청을 더 적은 수의 트랜잭션(Transaction)으로 결합할 수 있다. 분산된 메모리 접근(Scattered Memory Access)은 추가적인 트랜잭션을 발생시키고 대역폭을 낭비할 수 있다. 따라서 효율적인 CUDA 프로그램은 연속 접근(Contiguous Access)을 극대화하고 불필요한 전송을 최소화하도록 데이터 배치(Data Layout)와 스레드 인덱싱(Thread Indexing)을 구성한다.

전역 메모리 할당(Global Memory Allocation)과 전송 관리(Transfer Management)는 전체 애플리케이션 효율(Application Efficiency)에 영향을 미친다. CUDA 애플리케이션은 장치 버퍼(Device Buffer)를 명시적으로 할당하고 호스트 메모리(Host Memory)와 장치 메모리(Device Memory) 사이에서 데이터를 전송할 수 있다. 비동기 전송(Asynchronous Transfer)과 페이지 고정 호스트 메모리(Pinned Host Memory)는 적절하게 사용하면 전송 효율을 향상시킬 수 있다. 통합 메모리(Unified Memory)는 주소 공간 관리(Address-Space Management)를 단순화하지만, 접근 패턴을 적절하게 제어하지 않으면 페이지 이동(Page Migration)과 데이터 배치 결정(Placement Decision)으로 인한 오버헤드가 발생할 수 있다.

공유 메모리(Shared Memory)는 GPU 칩 내부에 위치하며 프로그래머가 관리하는 저지연 메모리 공간(Low-Latency Memory Space)이다. 공유 메모리는 동일한 스레드 블록(Thread Block) 내부의 스레드가 접근할 수 있으며, 중간 결과(Intermediate Result)를 교환하는 데 특히 유용하다. 전역 메모리와 달리 공유 메모리에는 재사용할 데이터를 명시적으로 적재할 수 있다. 제한된 용량으로 인해 신중한 할당이 필요하지만, 효과적으로 활용하면 반복적인 전역 메모리 접근(Global Memory Access)을 크게 줄일 수 있다.

공유 메모리 타일링(Shared-Memory Tiling)은 행렬 곱셈(Matrix Multiplication)과 합성곱(Convolution)을 위한 기본적인 최적화 기법(Optimization Technique)이다. 스레드는 협력하여 입력 데이터의 일부를 공유 메모리에 적재하고 여러 산술 연산(Arithmetic Operation)에서 재사용한다. 동기화(Synchronization)는 의존 연산(Dependent Computation)이 시작되기 전에 데이터가 준비되도록 보장한다. 타일링은 데이터 재사용(Data Reuse)을 증가시켜 외부 메모리 트래픽(External Memory Traffic)을 줄이고 산술 집약도(Arithmetic Intensity)를 향상시키며, 특히 연산 집약적인 신경망 커널(Compute-Intensive Neural Network Kernel)에서 효과적이다.

공유 메모리는 여러 접근 요청을 효율적으로 처리할 수 있도록 메모리 뱅크(Memory Bank)로 분할된다. 뱅크 충돌(Bank Conflict)은 워프(Warp) 내부의 스레드가 동일한 뱅크에 매핑된 서로 다른 주소를 요청할 때 발생하며, 일부 접근이 직렬화(Serialization)되어야 한다. 신중하게 선택한 인덱싱 방식(Indexing Scheme), 패딩(Padding), 메모리 배치(Memory Layout)는 이러한 충돌을 줄일 수 있다. 브로드캐스트(Broadcast)와 기타 지원되는 접근 패턴은 여러 스레드가 관련된 메모리 위치를 참조하더라도 충돌을 방지할 수 있다.

스레드가 중간 데이터를 공동으로 처리할 때 공유 메모리 동기화(Shared-Memory Synchronization)는 필수적이다. 블록 수준 배리어(Block-Level Barrier)는 참여 스레드가 의존 실행(Dependent Execution)을 계속하기 전에 필요한 메모리 연산(Memory Operation)을 완료하도록 보장한다. 잘못된 동기화는 경쟁 상태(Race Condition)나 일관되지 않은 결과(Inconsistent Result)를 발생시킬 수 있다. 과도한 동기화 역시 대기시간(Waiting Period)을 발생시켜 처리량(Throughput)을 감소시키므로 동기화 위치(Synchronization Placement)는 커널 최적화의 중요한 요소이다.

L2 캐시(L2 Cache)는 GPU 실행 자원(GPU Execution Resource)과 외부 전역 메모리(External Global Memory) 사이에 위치하는 중간 저장 계층(Intermediate Storage Layer)을 제공한다. 일반적으로 여러 스트리밍 멀티프로세서(Streaming Multiprocessor, SM)가 공유하며, 지원되는 메모리 경로(Memory Path)를 통해 접근한 데이터를 캐싱(Caching)한다. 캐시된 데이터에 반복적으로 접근하면 비용이 높은 외부 메모리 트랜잭션(External Memory Transaction)을 피할 수 있다. 따라서 L2 캐시는 작업 부하가 유용한 지역성(Locality)을 보일 때 유효 대역폭(Effective Bandwidth)을 높이고 평균 메모리 접근 지연시간(Average Memory Access Latency)을 줄인다.

L2 캐시 효율(L2 Cache Effectiveness)은 작업 집합 크기(Working-Set Size), 접근 지역성(Access Locality), 경쟁하는 메모리 트래픽(Competing Memory Traffic)에 따라 달라진다. 자주 재사용되는 텐서 데이터(Tensor Data)는 캐시에 유지될 수 있지만, 스트리밍 작업 부하(Streaming Workload)는 이전에 유용했던 데이터를 캐시에서 밀어낼 수 있다. 최신 CUDA 아키텍처는 지원되는 장치에서 접근 정책 메커니즘(Access-Policy Mechanism)을 포함한 일부 캐시 관리 기능(Cache-Management Capability)을 제공한다. 그러나 캐시 동작은 부분적으로 하드웨어가 관리하므로 애플리케이션 성능은 실측을 통해 평가해야 한다.

레지스터(Register)는 개별 GPU 스레드가 사용할 수 있는 가장 빠른 범용 저장 공간(General-Purpose Storage)을 나타낸다. 레지스터는 임시 변수(Temporary Variable), 주소(Address), 산술 피연산자(Arithmetic Operand), 중간 연산 결과(Intermediate Computation Result)를 저장한다. 레지스터는 스트리밍 멀티프로세서와 연결된 하드웨어 레지스터 파일(Hardware Register File)에서 할당된다. 공유 메모리와 달리 레지스터는 개별 스레드 전용(Thread-Private)이므로 스레드 내부 데이터 재사용(Thread-Local Data Reuse)이 많은 연산에서 특히 효과적이다.

레지스터 할당(Register Allocation)은 커널 점유율(Kernel Occupancy)과 스케줄링 유연성(Scheduling Flexibility)에 직접적인 영향을 미친다. 각 스트리밍 멀티프로세서는 제한된 레지스터 용량(Register Capacity)을 가지며, 스레드당 많은 레지스터가 필요한 커널은 상주 워프(Resident Warp)의 수를 감소시킬 수 있다. 낮은 점유율은 지연시간 은닉(Latency Hiding)을 제한할 수 있지만, 추가 레지스터는 메모리 트래픽을 줄여 성능을 향상시킬 수도 있다. 효과적인 최적화는 레지스터 사용량(Register Usage), 명령어 수준 병렬성(Instruction-Level Parallelism), 사용 가능한 실행 자원(Execution Resource)의 균형을 유지한다.

레지스터 스필링(Register Spilling)은 컴파일러(Compiler)가 필요한 모든 스레드 지역 값(Thread-Local Value)을 사용 가능한 레지스터에 유지할 수 없을 때 발생한다. 스필된 값(Spilled Value)은 일반적으로 로컬 메모리(Local Memory)에 저장되며, 로컬 메모리는 논리적으로 각 스레드에 전용이지만 물리적으로 장치 메모리(Device Memory)와 지원되는 캐시(Cache)를 기반으로 구현된다. 빈번한 스필링은 추가 메모리 연산을 발생시켜 성능을 크게 감소시킬 수 있다. 프로파일링(Profiling)과 컴파일러 진단(Compiler Diagnostics)은 과도한 레지스터 압박(Register Pressure)을 식별하는 데 도움이 된다.

레지스터와 공유 메모리 사이의 관계는 효율적인 CUDA 커널 설계(CUDA Kernel Design)의 핵심이다. 레지스터는 빠른 스레드 지역 연산(Thread-Local Computation)을 지원하고, 공유 메모리는 블록 내부의 협력(Cooperation)과 데이터 교환(Data Exchange)을 가능하게 한다. 전역 메모리는 저장 용량을 제공하며, L2 캐시는 일부 반복 접근의 비용을 줄인다. 효율적인 커널은 이러한 계층을 통해 데이터를 전략적으로 이동시키고, 자주 재사용되는 값을 가능한 한 실행 유닛(Execution Unit) 가까이에 유지한다.

메모리 지연시간(Memory Latency)과 대역폭(Bandwidth)은 서로 다른 성능 제한(Performance Limitation)을 발생시킨다. 지연시간은 개별 메모리 접근을 완료하는 데 필요한 시간을 의미하며, 대역폭은 단위 시간당 전송되는 데이터 양을 나타낸다. 불규칙하고 의존성이 있는 메모리 접근(Irregular Dependent Memory Access)을 수행하는 커널은 대역폭 활용률(Bandwidth Utilization)이 낮더라도 지연시간에 의해 성능이 제한될 수 있다. 반대로 대규모 스트리밍 연산(Large Streaming Operation)은 개별 접근 지연을 은닉할 충분한 병렬성을 확보하더라도 메모리 대역폭을 포화(Saturation)시킬 수 있다.

산술 집약도(Arithmetic Intensity)는 메모리 동작(Memory Behavior)과 연산 처리량(Computational Throughput)을 연결한다. 산술 집약도는 데이터 이동량(Data Movement)에 대한 산술 연산량(Arithmetic Work)의 비율을 측정하며, 일반적으로 바이트당 연산 수(Operations per Byte)로 표현된다. 산술 집약도가 낮은 커널은 메모리 대역폭에 의해 제한되는 경우가 많지만, 데이터 재사용이 충분한 커널은 연산 성능에 의해 제한되는 연산 병목(Compute-Bound) 상태가 될 수 있다. 루프라인 모델(Roofline Model)은 이러한 관계를 설명하고 최적화가 메모리 트래픽과 산술 실행(Arithmetic Execution) 중 어디에 집중되어야 하는지 식별하는 데 도움이 된다.

메모리 계층 구조 최적화(Memory Hierarchy Optimization)는 딥러닝 작업 부하(Deep Learning Workload)에서 특히 중요하다. 행렬 곱셈, 합성곱, 어텐션 연산(Attention Operation)은 대규모 텐서와 중간 결과에 반복적으로 접근한다. 효율적인 구현은 레지스터와 공유 메모리를 활용하여 데이터 재사용을 늘리고, L2 캐시를 통해 외부 메모리 트래픽을 줄인다. 커널 융합(Kernel Fusion)은 연속된 연산 사이에서 불필요한 중간 데이터 쓰기(Intermediate Write)와 읽기(Read)를 제거하여 성능을 추가로 향상시킬 수 있다.

트랜스포머 모델(Transformer Model)은 가중치(Weight), 활성화 값(Activation), 어텐션 점수(Attention Score), 키-값 캐시(Key-Value Cache, KV Cache)를 포함하는 복잡한 메모리 접근 요구사항(Memory-Access Requirement)을 생성한다. 추론(Inference) 과정에서는 특히 긴 문맥(Long Context)이나 큰 배치(Large Batch)에서 키-값 캐시 트래픽이 주요 대역폭 제약조건(Bandwidth Constraint)이 될 수 있다. 최적화된 어텐션 알고리즘(Optimized Attention Algorithm)은 타일링(Tiling)과 융합(Fusion)을 통해 중간 저장 공간을 줄이고 지역성을 향상시킨다. 이러한 기법의 효과는 시퀀스 길이(Sequence Length), 텐서 차원(Tensor Dimension), 사용 가능한 메모리 용량에 따라 달라진다.

합성곱 신경망(Convolutional Neural Network, CNN)은 채널(Channel)과 공간 차원(Spatial Dimension)에 걸친 효율적인 접근을 지원하는 메모리 배치(Memory Layout)의 이점을 얻는다. 서로 다른 텐서 배치(Tensor Layout)는 메모리 접근 병합(Memory Coalescing), 캐시 지역성(Cache Locality), 특수 행렬 명령어(Specialized Matrix Instruction)와의 호환성에 영향을 미칠 수 있다. cuDNN과 같은 라이브러리는 텐서 형태(Tensor Shape)와 하드웨어 특성(Hardware Characteristic)에 따라 알고리즘을 선택한다. 성능 향상은 메모리 구성과 산술 실행을 독립적으로 최적화하기보다 서로 조정함으로써 얻어지는 경우가 많다.

GPU 메모리 용량(GPU Memory Capacity)은 대규모 인공지능 모델(Large Artificial Intelligence Model)의 중요한 제약조건이다. 학습(Training) 과정에서 모델 매개변수(Model Parameter), 옵티마이저 상태(Optimizer State), 기울기(Gradient), 활성화 값(Activation)은 사용 가능한 장치 메모리(Device Memory)를 공유하며 경쟁한다. 활성화 체크포인팅(Activation Checkpointing), 혼합 정밀도(Mixed Precision), 텐서 분할(Tensor Partitioning)과 같은 기법은 서로 다른 연산 비용(Computational Cost)을 수반하면서 메모리 요구량을 줄인다. 효율적인 메모리 계획(Memory Planning)은 추가 연산 및 통신 오버헤드(Communication Overhead)와 균형을 유지하면서 메모리 할당 실패(Allocation Failure)를 방지하는 데 도움이 된다.

메모리 단편화(Memory Fragmentation)와 할당 오버헤드(Allocation Overhead)는 텐서를 반복적으로 생성하고 해제하는 애플리케이션에 영향을 미칠 수 있다. 프레임워크 메모리 할당기(Framework Memory Allocator)는 일반적으로 이전에 할당한 버퍼(Buffer)를 재사용하고 비용이 높은 메모리 할당 연산을 줄이기 위해 캐싱 전략(Caching Strategy)을 사용한다. 그러나 단편화된 메모리 영역(Fragmented Memory Region)과 예측하기 어려운 할당 패턴(Unpredictable Allocation Pattern)은 유효 메모리 용량(Effective Memory Capacity)을 제한할 수 있다. 안정적인 텐서 형태(Stable Tensor Shape), 버퍼 재사용(Buffer Reuse), 신중한 수명 관리(Lifetime Management)는 운영 효율(Operational Efficiency)을 향상시킬 수 있다.

비동기 메모리 이동(Asynchronous Memory Movement)은 하드웨어 및 의존성 요구사항(Dependency Requirement)이 허용하는 경우 데이터 전송(Data Transfer)과 연산(Computation)을 중첩할 수 있다. CUDA 스트림(CUDA Stream), 비동기 복사 연산(Asynchronous Copy Operation), 아키텍처별 메커니즘(Architecture-Specific Mechanism)은 파이프라인 실행(Pipelined Execution)을 지원한다. 이중 버퍼링(Double Buffering)은 하나의 데이터 타일(Data Tile)을 처리하는 동안 다른 타일을 준비하도록 한다. 이러한 기법은 산술 유닛(Arithmetic Unit)이 피연산자(Operand)를 기다리는 시간을 줄여 활용률(Utilization)을 향상시키지만, 동기화와 자원 사용량(Resource Usage)을 신중하게 제어해야 한다.

프로파일링(Profiling)은 메모리 계층 구조의 병목(Memory Hierarchy Bottleneck)을 식별하는 데 필요한 실측 근거를 제공한다. NVIDIA 엔사이트 컴퓨트(NVIDIA Nsight Compute)는 캐시 적중률(Cache Hit Rate), 메모리 처리량(Memory Throughput), 메모리 트랜잭션(Memory Transaction), 레지스터 압박(Register Pressure), 공유 메모리 충돌(Shared-Memory Conflict)과 관련된 지표를 제공한다. 이러한 측정값은 대역폭 포화(Bandwidth Saturation)를 지연시간 제한(Latency Limitation)이나 비효율적인 접근 패턴(Inefficient Access Pattern)과 구분하는 데 도움이 된다. 최적화는 모든 커널이 동일한 캐싱 또는 타일링 전략의 이점을 얻는다고 가정하기보다 관찰된 병목에 집중해야 한다.

GPU 메모리 계층 구조는 궁극적으로 연산 자원(Computational Resource)이 데이터를 얼마나 효율적으로 공급받고 재사용할 수 있는지를 결정한다. 전역 메모리(Global Memory)는 저장 용량을 제공하고, L2 캐시(L2 Cache)는 반복적인 외부 메모리 접근을 줄이며, 공유 메모리(Shared Memory)는 협력적 데이터 재사용(Cooperative Data Reuse)을 가능하게 하고, 레지스터(Register)는 빠른 스레드 지역 연산(Thread-Local Operation)을 지원한다. 이러한 상호 보완적 역할(Complementary Role)을 이해하면 개발자는 처리량을 높이고 지연시간을 줄이면서 딥러닝(Deep Learning), 시뮬레이션(Simulation), 로보틱스(Robotics) 작업 부하를 효율적으로 지원하는 확장 가능한 CUDA 커널(Scalable CUDA Kernel)을 설계할 수 있다.

## 01.07 GPU Architectures

![](images/image8.png){width="7.268055555555556in" height="7.268055555555556in"}

GPU 아키텍처(GPU Architecture)는 병렬 연산(Parallel Computation), 메모리 대역폭(Memory Bandwidth), 인공지능 가속(Artificial Intelligence Acceleration)에 대한 증가하는 요구를 충족하기 위해 발전해 왔다. NVIDIA 볼타(Volta), 암페어(Ampere), 에이다(Ada), 호퍼(Hopper)는 서로 다른 설계 우선순위와 기능을 가진 연속적인 아키텍처 발전 단계를 나타낸다. 이러한 발전 과정은 GPU 설계가 그래픽 처리(Graphics Processing)를 넘어 특수 행렬 연산(Specialized Matrix Computation), 효율적인 데이터 이동(Data Movement), 대규모 인공지능 작업 부하(Large-Scale AI Workload)로 확장되어 왔음을 보여준다.

NVIDIA 볼타(Volta)는 스트리밍 멀티프로세서(Streaming Multiprocessor, SM)에 텐서 코어(Tensor Core)를 통합함으로써 중요한 아키텍처 전환(Architectural Transition)을 이루었다. 볼타 V100(Volta V100)은 기존 CUDA 코어(CUDA Core)와 특수 행렬 실행 하드웨어(Specialized Matrix Execution Hardware)를 결합하여 혼합 정밀도 연산(Mixed-Precision Computation)을 가속하였다. 이러한 설계는 딥러닝 학습(Deep Learning Training)과 과학 응용(Scientific Application)의 성능을 크게 향상시켰다. 볼타는 텐서 코어 가속을 현대 NVIDIA 연산 아키텍처(Compute Architecture)의 핵심 기능으로 확립하였다.

볼타는 독립적 스레드 스케줄링(Independent Thread Scheduling)을 도입하여 워프(Warp) 내부의 실행을 더욱 유연하게 관리할 수 있도록 하였다. 이전 아키텍처는 워프 수준의 동기 실행 가정(Warp-Level Lockstep Assumption)에 더 크게 의존했지만, 볼타는 더욱 세분화된 스레드 실행 상태(Thread Execution State)를 유지하였다. 이러한 변화는 분기 제어 흐름(Divergent Control Flow)과 일부 동기화 패턴(Synchronization Pattern)에 대한 지원을 향상시켰다. 그러나 스레드가 데이터를 교환하거나 조정된 실행(Coordinated Execution)에 의존하는 경우 올바른 CUDA 프로그램은 여전히 명시적 동기화(Explicit Synchronization)를 필요로 한다.

볼타 메모리 하위 시스템(Volta Memory Subsystem)은 고대역폭 메모리(High-Bandwidth Memory), 온칩 캐시(On-Chip Cache), 구성 가능한 공유 메모리 자원(Configurable Shared-Memory Resource)을 결합하였다. V100은 HBM2 메모리와 상당한 용량의 L2 캐시(L2 Cache)를 사용하여 데이터 집약적인 과학 연산(Data-Intensive Scientific Computation)과 신경망 작업 부하(Neural Network Workload)를 지원하였다. 개선된 메모리 접근 기능(Memory Access Capability)은 텐서 코어의 증가된 산술 처리량(Arithmetic Throughput)을 보완하였다. 그러나 효율적인 애플리케이션을 구현하려면 데이터 지역성(Data Locality), 대역폭, 레지스터 압박(Register Pressure)을 신중하게 관리해야 했다.

NVIDIA 암페어(Ampere)는 텐서 코어 기능(Tensor Core Functionality)을 확장하고 연산과 메모리 접근 사이의 균형을 개선하였다. A100은 3세대 텐서 코어(Third-Generation Tensor Core)를 도입하여 기존 혼합 정밀도 연산과 함께 TF32 및 BF16과 같은 수치 형식(Numerical Format)을 지원하였다. TF32는 데이터 저장 형식을 저정밀도로 완전히 변환하지 않고도 일부 FP32 행렬 작업 부하(Matrix Workload)를 가속하였다. 이러한 기능으로 암페어는 대규모 인공지능 학습(Large-Scale AI Training)에서 특히 중요한 아키텍처가 되었다.

암페어는 일부 텐서 코어 연산에서 구조적 희소성(Structured Sparsity)을 위한 하드웨어 지원도 도입하였다. 호환 가능한 희소 행렬(Sparse Matrix)은 필요한 산술 연산량(Arithmetic Work)을 줄여 지원되는 조건에서 유효 처리량(Effective Throughput)을 높일 수 있었다. 또한 이 아키텍처는 비동기 데이터 이동 기능(Asynchronous Data Movement Capability)을 확장하여 최적화된 커널(Optimized Kernel)이 연산을 조정하면서 데이터를 공유 메모리(Shared Memory)로 전송할 수 있도록 하였다. 이러한 개선은 행렬 실행(Matrix Execution)과 메모리 계층 구조 최적화(Memory Hierarchy Optimization) 사이의 연관성을 강화하였다.

암페어 A100은 HBM2e 메모리, 볼타 V100보다 큰 L2 캐시, 다중 GPU 컴퓨팅(Multi-GPU Computing)을 위한 고속 인터커넥트(High-Speed Interconnect) 기능을 통합하였다. 이 아키텍처는 대규모 행렬과 상당한 통신량을 포함하는 까다로운 학습 작업 부하(Training Workload)를 지원하였다. 다중 인스턴스 GPU(Multi-Instance GPU, MIG) 기능을 통해 지원되는 A100 장치의 자원을 격리된 컴퓨팅 인스턴스(Isolated Computing Instance)로 분할할 수 있었다. 이러한 기능은 공유 인프라(Shared Infrastructure)와 다양한 배포 요구사항(Deployment Requirement)에 대한 유연성을 향상시켰다.

암페어는 데이터센터 GPU(Data-Center GPU)와 소비자용 GPU(Consumer GPU)를 포함한 여러 NVIDIA 제품군(Product Family)에 적용되었다. 이러한 제품들은 공통된 아키텍처 기반(Architectural Foundation)을 공유했지만 메모리 기술(Memory Technology), 텐서 코어 기능, 수치 연산 기능(Numerical Capability)은 서로 달랐다. A100은 과학 컴퓨팅(Scientific Computing)과 대규모 학습에 중점을 두었으며, 다른 암페어 제품은 그래픽, 시각화(Visualization), 범용 가속(General Acceleration)을 목표로 하였다. 따라서 하드웨어 선택에서는 아키텍처 이름만이 아니라 구체적인 장치 기능(Device Capability)을 확인해야 했다.

NVIDIA 에이다 러브레이스(Ada Lovelace)는 그래픽, 렌더링(Rendering), 인공지능 지원 작업 부하(AI-Assisted Workload)에 특히 중점을 두어 GPU 아키텍처를 개선하였다. 에이다 GPU는 4세대 텐서 코어(Fourth-Generation Tensor Core)를 도입하고 지원되는 신경망 연산(Neural Network Operation)의 성능을 향상시켰다. 또한 이 아키텍처는 광선 추적(Ray Tracing)의 발전을 포함하여 특수 그래픽 기능(Specialized Graphics Capability)을 확장하였다. 이러한 개선으로 에이다는 시각 처리(Visual Processing), 신경망 추론(Neural Inference), 대화형 연산(Interactive Computation)을 결합하는 애플리케이션에 적합해졌다.

에이다는 지원되는 제품 전반에서 캐시 용량(Cache Capacity)과 메모리 효율(Memory Efficiency)을 개선하였다. 많은 에이다 GPU의 더 큰 L2 캐시는 유용한 지역성(Locality)을 보이는 작업 부하에서 외부 메모리 트래픽(External Memory Traffic)을 줄였다. 이러한 설계는 일부 애플리케이션의 대역폭 제약조건(Bandwidth Constraint)을 완화하는 데 도움이 되었다. 실제 성능은 메모리 접근 패턴(Memory Access Pattern), 캐시 동작(Cache Behavior), 연산 처리량(Computational Throughput)과 외부 메모리 대역폭 사이의 관계에 따라 달라졌다.

에이다 텐서 코어(Ada Tensor Core)는 가속된 행렬 연산(Accelerated Matrix Computation)과 지원되는 장치의 FP8 기능을 포함한 일부 저정밀도 연산(Low-Precision Operation)을 지원하였다. 이러한 기능은 호환 가능한 소프트웨어 실행 경로(Software Execution Path)가 제공되는 경우 인공지능 추론(AI Inference)과 기타 텐서 집약적 애플리케이션(Tensor-Intensive Application)에 이점을 제공하였다. 그러나 에이다 제품의 기능은 모델에 따라 달랐으며 모든 기능이 동일한 실제 성능을 제공하는 것은 아니었다. 수치 형식 지원(Numerical Format Support)과 소프트웨어 호환성(Software Compatibility)은 여전히 중요한 배포 고려사항이었다.

에이다 GPU는 전력 및 시스템 제약조건(Power and System Constraint)이 중요한 워크스테이션 인공지능(Workstation AI), 시각 컴퓨팅(Visual Computing), 엣지 지향 가속(Edge-Oriented Acceleration)에 폭넓게 활용될 수 있었다. 컴퓨터 비전(Computer Vision), 신경망 렌더링(Neural Rendering), 로보틱스 인식(Robotics Perception)과 같은 애플리케이션은 개선된 텐서 처리(Tensor Processing)와 캐시 효율의 이점을 얻을 수 있었다. 그러나 소비자용 및 전문가용 에이다 제품은 메모리 용량, 신뢰성 기능(Reliability Feature), 배포 특성(Deployment Characteristic)에서 차이가 있었다. 따라서 시스템 설계에서는 전체 하드웨어 구성(Hardware Configuration)을 평가해야 했다.

NVIDIA 호퍼(Hopper)는 대규모 인공지능과 고성능 컴퓨팅(High-Performance Computing, HPC)을 향한 또 다른 중요한 전환을 나타냈다. H100은 확장된 수치 연산 기능과 특수 실행 메커니즘(Specialized Execution Mechanism)을 갖춘 4세대 텐서 코어를 도입하였다. 호퍼는 트랜스포머 작업 부하(Transformer Workload), 혼합 정밀도 학습(Mixed-Precision Training), 대규모 행렬의 효율적인 처리에 중점을 두었다. 이 아키텍처는 산술 유닛(Arithmetic Unit), 메모리 이동, 동기화, 인터커넥트 성능(Interconnect Performance) 전반의 개선을 통합하였다.

호퍼는 지원되는 FP8 및 고정밀도 연산(Higher-Precision Operation)을 조정하여 트랜스포머 연산(Transformer Computation)을 가속하는 트랜스포머 엔진(Transformer Engine)을 도입하였다. 이 엔진은 소프트웨어와 하드웨어 기능을 활용하여 지원되는 작업 부하에 적합한 수치 표현(Numerical Representation)을 선택한다. 이러한 접근 방식은 허용 가능한 수치적 동작(Numerical Behavior)을 유지하면서 메모리 요구량(Memory Requirement)을 줄이고 처리량을 향상시킬 수 있다. 효과는 모델 특성(Model Characteristic), 스케일링 전략(Scaling Strategy), 프레임워크 통합(Framework Integration)에 따라 달라진다.

호퍼는 텐서 메모리 가속기(Tensor Memory Accelerator, TMA)와 워프 그룹 행렬 연산(Warp-Group Matrix Operation)과 같은 메커니즘을 통해 비동기 실행(Asynchronous Execution)을 확장하였다. 이러한 기능은 다차원 텐서 데이터(Multidimensional Tensor Data)의 효율적인 이동과 조정된 행렬 연산(Coordinated Matrix Computation)을 지원한다. 최적화된 커널은 데이터 준비(Data Preparation)와 산술 실행(Arithmetic Execution)을 더욱 효과적으로 중첩할 수 있다. 그러나 이러한 프로그래밍 모델(Programming Model)은 올바르게 처리해야 하는 추가적인 동기화 및 자원 관리 요구사항(Resource-Management Requirement)을 수반한다.

호퍼 H100은 고대역폭 메모리(High-Bandwidth Memory)와 상당한 용량의 L2 캐시를 사용하여 집약적인 학습 및 과학 연산 작업 부하를 지원한다. 메모리 하위 시스템(Memory Subsystem)은 특수 행렬 하드웨어(Specialized Matrix Hardware)에 높은 속도로 데이터를 공급하도록 설계되었다. 개선된 인터커넥트 기술(Interconnect Technology)은 대규모 시스템에서 GPU 사이의 통신을 지원한다. 그러나 분산 처리 성능(Distributed Performance)은 여전히 통신 패턴(Communication Pattern), 모델 분할(Model Partitioning), 연산과 데이터 교환 사이의 균형에 따라 달라진다.

볼타에서 호퍼까지의 아키텍처 발전 과정(Architectural Progression)은 수치 연산 특화(Numerical Specialization)의 중요성이 증가하고 있음을 보여준다. 볼타는 텐서 코어 연산의 기반을 확립하였고, 암페어는 정밀도 형식(Precision Format)과 희소성 지원(Sparsity Support)을 확장하였으며, 에이다는 그래픽 중심의 인공지능 기능(Graphics-Oriented AI Capability)을 개선하였다. 호퍼는 고급 트랜스포머 가속(Advanced Transformer Acceleration)을 도입하였다. 각 세대는 연산, 메모리, 스케줄링 사이의 상호작용을 개선하였으며, 성능 향상은 소프트웨어가 대상 아키텍처의 기능을 활용하는 정도에 따라 달라졌다.

메모리 계층 구조 설계(Memory Hierarchy Design)는 산술 연산 기능(Arithmetic Capability)과 함께 발전하였다. 더 큰 캐시, 개선된 공유 메모리 메커니즘(Shared-Memory Mechanism), 효율적인 데이터 이동은 외부 메모리 시스템(External Memory System)에 가해지는 부담을 줄였다. 이러한 변화는 데이터 재사용(Data Reuse)이 많은 행렬 곱셈(Matrix Multiplication), 어텐션(Attention), 합성곱(Convolution) 작업 부하에서 특히 유용하였다. 그러나 메모리 용량과 대역폭은 여전히 중요한 제약조건이며, 특히 모델 크기(Model Size)가 사용 가능한 장치 자원(Device Resource)을 초과하는 경우 더욱 중요하다.

CUDA 소프트웨어 호환성(CUDA Software Compatibility)은 애플리케이션이 여러 GPU 세대를 대상으로 실행될 수 있도록 하지만 성능 이식성(Performance Portability)이 자동으로 보장되는 것은 아니다. 하나의 아키텍처에 최적화된 커널은 레지스터 용량(Register Capacity), 캐시 구성(Cache Organization), 명령어 지원(Instruction Support), 스케줄링 동작(Scheduling Behavior)의 변화로 인해 다른 아키텍처에서 서로 다른 성능을 보일 수 있다. 컴파일러 옵션(Compiler Option)과 라이브러리 버전(Library Version)은 활용되는 하드웨어 기능에 영향을 미친다. 최대 성능을 얻기 위해서는 아키텍처 인식 튜닝(Architecture-Aware Tuning)이 여전히 중요하다.

텐서 코어 프로그래밍(Tensor Core Programming)은 이러한 이식성 문제(Portability Challenge)를 잘 보여준다. GPU 세대마다 지원하는 행렬 명령어 형태(Matrix Instruction Form), 피연산자 형식(Operand Format), 실행 메커니즘(Execution Mechanism)이 다르다. 볼타의 워프 수준 행렬 연산(Warp-Level Matrix Operation)을 위해 설계된 커널이 호퍼의 모든 비동기 기능(Asynchronous Capability)을 자동으로 활용할 수 있는 것은 아니다. 상위 수준 라이브러리(High-Level Library)와 컴파일러 프레임워크(Compiler Framework)는 일부 아키텍처 차이를 숨길 수 있지만, 특수 커널(Specialized Kernel)은 별도의 최적화 경로(Optimization Path)가 필요한 경우가 많다.

다중 GPU 확장성(Multi-GPU Scalability)은 개별 GPU의 연산 처리량뿐 아니라 다른 아키텍처 기능에도 의존한다. 고속 인터커넥트(High-Speed Interconnect), 메모리 용량, 통신 라이브러리(Communication Library), 시스템 토폴로지(System Topology)는 분산 학습 성능(Distributed Training Performance)에 영향을 미친다. 데이터 병렬성(Data Parallelism)은 효율적인 기울기 동기화(Gradient Synchronization)를 요구하며, 텐서 병렬성(Tensor Parallelism)과 파이프라인 병렬성(Pipeline Parallelism)은 추가적인 통신 의존성(Communication Dependency)을 발생시킨다. 호퍼와 암페어 데이터센터 시스템은 이러한 까다로운 연산 환경(Computational Environment)을 지원하도록 설계되었다.

에너지 효율(Energy Efficiency)과 열 특성(Thermal Characteristic)은 GPU 세대를 비교할 때 필수적으로 고려해야 하는 요소이다. 아키텍처 개선은 와트당 연산량(Operations per Watt)을 증가시킬 수 있지만 실제 효율은 정밀도(Precision), 작업 부하 활용률(Workload Utilization), 메모리 트래픽(Memory Traffic), 전력 제한(Power Limit)에 따라 달라진다. 데이터센터 GPU는 지속적인 연산(Sustained Computation)과 인프라 통합(Infrastructure Integration)을 우선시하는 반면, 워크스테이션 제품(Workstation Product)은 서로 다른 운영 제약조건(Operational Constraint) 아래에서 성능의 균형을 유지한다. 따라서 하드웨어 비교에서는 최대 산술 성능 사양(Peak Arithmetic Specification)만이 아니라 측정된 애플리케이션 효율(Measured Application Efficiency)을 고려해야 한다.

딥러닝 학습(Deep Learning Training)에서 볼타와 암페어는 혼합 정밀도 가속(Mixed-Precision Acceleration)의 중요한 기반을 확립하였으며, 호퍼는 트랜스포머 중심 실행(Transformer-Oriented Execution)에 대한 지원을 확장하였다. 에이다는 호환 가능한 추론(Inference) 및 시각 인공지능 작업 부하(Visual AI Workload)에 강력한 기능을 제공하지만 배포 적합성(Deployment Suitability)은 제품마다 다르다. 모델 크기, 수치 안정성(Numerical Stability), 배치 구성(Batch Configuration), 메모리 요구사항은 각 아키텍처의 실제 이점을 결정한다. 신뢰성 높은 선택을 위해서는 대표적인 작업 부하(Representative Workload)를 사용한 벤치마킹(Benchmarking)이 필요하다.

로보틱스(Robotics), 시뮬레이션(Simulation), 자율 시스템(Autonomous System)은 GPU의 순수 연산 처리량(Raw GPU Throughput) 이외에도 추가적인 요구사항을 가진다. 실시간 인식(Real-Time Perception)은 가속된 신경망 추론(Accelerated Neural Inference)의 이점을 얻으며, 시뮬레이션은 상당한 메모리 대역폭과 수치 정확도(Numerical Accuracy)를 요구할 수 있다. 전력 소비(Power Consumption), 열 제한(Thermal Limit), 소프트웨어 호환성, 예측 가능한 실행 지연시간(Predictable Execution Latency)은 하드웨어 선택에 영향을 미친다. 가장 적합한 아키텍처는 단순히 최신 세대인지 여부가 아니라 전체 시스템 작업 부하(Complete System Workload)에 따라 결정된다.

볼타, 암페어, 에이다, 호퍼를 이해하는 것은 현대 인공지능 시스템에서 NVIDIA GPU 기능을 평가하기 위한 기반을 제공한다. 이러한 아키텍처의 발전은 텐서 코어(Tensor Core), 메모리 계층 구조(Memory Hierarchy), 실행 스케줄링(Execution Scheduling), 수치 정밀도(Numerical Precision), 다중 GPU 통신(Multi-GPU Communication)의 조화로운 개선을 반영한다. 효과적인 배포(Effective Deployment)를 위해서는 이러한 아키텍처 특성(Architectural Characteristic)을 애플리케이션 요구사항(Application Requirement)에 맞추어야 한다. 이러한 하드웨어-소프트웨어 정합(Hardware-Software Alignment)은 확장 가능한 학습(Scalable Training), 효율적인 추론(Efficient Inference), 신뢰성 높은 연산 성능(Reliable Computational Performance)을 가능하게 한다.

## 01.08 Optimization Strategies [w/Code]

![](images/image9.png){width="7.268055555555556in" height="7.268055555555556in"}

GPU 최적화 전략(GPU Optimization Strategy)은 메모리 지연시간(Memory Latency), 데이터 이동(Data Movement), 자원 경합(Resource Contention)을 최소화하면서 연산 처리량(Computational Throughput)을 극대화하는 것을 목표로 한다. 효과적인 최적화를 위해서는 CUDA 커널(CUDA Kernel)이 스트리밍 멀티프로세서(Streaming Multiprocessor, SM), 워프(Warp), 레지스터(Register), 캐시(Cache), 실행 파이프라인(Execution Pipeline)과 어떻게 상호작용하는지 이해해야 한다. 성능은 산술 연산 능력(Arithmetic Capability)뿐만 아니라 명령어와 데이터가 실행 유닛(Execution Unit)에 얼마나 효율적으로 전달되는지에 따라서도 결정된다. 체계적인 측정(Systematic Measurement)은 필수적이다.

성능 최적화(Performance Optimization)는 작업 부하(Workload)에서 지배적인 병목(Bottleneck)을 식별하는 것에서 시작한다. 연산 병목 커널(Compute-Bound Kernel)은 주로 산술 실행 용량(Arithmetic Execution Capacity)에 의해 제한되며, 메모리 병목 커널(Memory-Bound Kernel)은 대역폭(Bandwidth)이나 접근 지연시간(Access Latency)에 의존한다. 다른 커널은 병렬성 부족(Insufficient Parallelism), 동기화 오버헤드(Synchronization Overhead), 명령어 의존성(Instruction Dependency)으로 인해 성능이 저하될 수 있다. NVIDIA 엔사이트 컴퓨트(NVIDIA Nsight Compute)와 같은 프로파일링 도구(Profiling Tool)는 최적화 결정을 내리기 전에 이러한 상태를 구분하는 데 필요한 측정값을 제공한다.

메모리 접근 병합(Memory Coalescing)은 GPU 효율을 향상시키는 가장 중요한 기법 중 하나이다. 워프 내부의 스레드(Thread)는 가능한 한 연속되거나 적절하게 정렬된 메모리 위치(Aligned Memory Location)에 접근해야 한다. 병합된 메모리 접근(Coalesced Memory Access)은 유효 데이터를 전송하는 데 필요한 전역 메모리 트랜잭션(Global Memory Transaction)의 수를 줄인다. 비효율적인 접근 패턴(Access Pattern)은 대역폭 소비와 지연시간을 증가시킨다. 데이터 배치 변환(Data Layout Transformation)과 스레드 인덱싱(Thread Indexing)의 재구성을 통해 메모리 효율(Memory Efficiency)을 크게 향상시킬 수 있다.

공유 메모리 타일링(Shared-Memory Tiling)은 재사용 가능한 데이터를 실행 유닛 가까이에 저장하여 반복적인 전역 메모리 접근(Global Memory Access)을 줄인다. 행렬 곱셈 커널(Matrix Multiplication Kernel)은 일반적으로 입력 행렬(Input Matrix)을 타일(Tile)로 나누고 협력하는 스레드가 이를 공유 메모리(Shared Memory)에 적재한다. 이러한 타일은 교체되기 전에 여러 산술 연산(Arithmetic Operation)에서 재사용된다. 적절한 동기화(Synchronization)는 정확성을 유지하며, 신중하게 설계된 메모리 배치(Memory Layout)는 뱅크 충돌(Bank Conflict)을 최소화하고 유효 대역폭(Effective Bandwidth)을 향상시킨다.

레지스터 최적화(Register Optimization)는 빠른 스레드 지역 저장 공간(Thread-Local Storage)과 하드웨어 자원 제한(Hardware Resource Limitation) 사이의 균형을 유지한다. 레지스터는 임시 값(Temporary Value)에 대한 저지연 접근(Low-Latency Access)을 제공하지만, 과도한 할당은 상주 워프(Resident Warp)의 수를 감소시킨다. 레지스터 스필링(Register Spilling)은 추가적인 로컬 메모리 트래픽(Local-Memory Traffic)을 발생시켜 처리량을 크게 감소시킬 수 있다. 개발자는 필요한 데이터 재사용(Data Reuse)을 유지하면서 커널 구조(Kernel Structure), 변수 수명(Variable Lifetime), 실행 구성(Launch Configuration)을 조정하여 레지스터 압박(Register Pressure)을 제어할 수 있다.

점유율(Occupancy)은 스트리밍 멀티프로세서에서 사용 가능한 워프 수용 능력(Warp Capacity) 중 활성 워프(Active Warp)가 차지하는 비율을 나타낸다. 높은 점유율은 추가적인 스케줄링 기회(Scheduling Opportunity)를 제공하여 지연시간 은닉(Latency Hiding)을 향상시킬 수 있다. 그러나 최대 점유율(Maximum Occupancy)이 반드시 최대 성능(Maximum Performance)을 의미하지는 않는다. 효율적인 데이터 재사용과 충분한 명령어 수준 병렬성(Instruction-Level Parallelism)을 가진 커널은 중간 수준의 점유율과 더 많은 레지스터 가용성(Register Availability)을 통해 더 높은 성능을 달성할 수 있다.

워프 분기 발산(Warp Divergence)은 동일한 워프 내부의 스레드가 서로 다른 제어 경로(Control Path)를 따를 때 실행 효율(Execution Efficiency)을 감소시킨다. 불규칙한 데이터(Irregular Data)를 포함하는 조건 분기(Conditional Branch)는 하드웨어가 서로 다른 활성 스레드 마스크(Active-Thread Mask)를 사용하여 별도의 명령어 경로(Instruction Path)를 실행하도록 만들 수 있다. 인접한 스레드가 유사한 연산을 수행하도록 작업 부하를 구성하면 분기 발산을 줄일 수 있다. 조건부 실행(Predication)과 분기 구조 재구성(Branch Restructuring)이 도움이 될 수 있지만, 추가 명령어로 인한 비용이 이점을 상쇄할 수도 있다.

스레드 블록 구성(Thread-Block Configuration)은 스케줄링 유연성(Scheduling Flexibility), 자원 소비(Resource Consumption), 실행 효율에 영향을 미친다. 블록 크기(Block Size)는 일반적으로 하드웨어 워프 크기(Warp Size)에 맞추면서 레지스터와 공유 메모리 요구사항을 고려해야 한다. 지나치게 큰 블록은 동시 상주(Concurrent Residency)를 제한할 수 있으며, 지나치게 작은 블록은 스케줄링 오버헤드(Scheduling Overhead)를 증가시키거나 협력 효율(Cooperation Efficiency)을 감소시킬 수 있다. 최적 구성은 커널과 GPU 아키텍처(GPU Architecture)에 따라 달라지므로 실험적 튜닝(Empirical Tuning)이 필요하다.

명령어 수준 병렬성(Instruction-Level Parallelism)은 스레드나 워프가 이전 결과를 기다리지 않고 독립적인 연산(Independent Operation)을 실행할 수 있도록 한다. 독립적인 산술 연산(Arithmetic Work)을 늘리면 의존성으로 인한 실행 정지(Dependency Stall)를 줄이고 파이프라인 활용률(Pipeline Utilization)을 향상시킬 수 있다. 루프 펼치기(Loop Unrolling)는 추가적인 병렬성을 노출하고 제어 오버헤드(Control Overhead)를 줄이는 경우가 있다. 그러나 과도한 루프 펼치기는 코드 크기(Code Size)와 레지스터 사용량을 증가시켜 점유율과 명령어 캐시 효율(Instruction-Cache Efficiency)을 감소시킬 수 있다.

커널 융합(Kernel Fusion)은 여러 연산 단계(Computational Stage)를 더 적은 수의 GPU 커널 실행(Kernel Launch)으로 결합한다. 커널 융합은 중간 데이터의 전역 메모리 쓰기(Global-Memory Write)와 읽기(Global-Memory Read)를 제거하여 데이터 이동과 실행 오버헤드(Launch Overhead)를 줄일 수 있다. 활성화 함수(Activation Function), 정규화(Normalization), 요소별 변환(Elementwise Transformation)은 대표적인 융합 대상이다. 그러나 과도한 융합은 레지스터 압박을 증가시키거나 스케줄링을 복잡하게 만들 수 있으므로 효과는 최종 커널 구조에 따라 달라진다.

텐서 코어 최적화(Tensor Core Optimization)는 행렬 연산(Matrix Operation)을 지원되는 하드웨어 명령어 형태(Hardware Instruction Shape)와 수치 형식(Numerical Format)에 매핑하는 데 중점을 둔다. FP16, BF16, TF32 및 일부 FP8 연산은 호환 가능한 아키텍처에서 상당한 가속 효과를 제공할 수 있다. 효율적인 구현은 적절한 텐서 배치(Tensor Layout), 정렬된 차원(Aligned Dimension), 충분한 산술 집약도(Arithmetic Intensity)를 활용한다. 수치 안정성(Numerical Stability)과 누산 정밀도(Accumulation Precision)는 처리량 향상과 함께 평가해야 한다.

비동기 실행(Asynchronous Execution)은 연산(Computation)과 메모리 이동(Memory Movement)을 중첩할 수 있도록 한다. CUDA 스트림(CUDA Stream)은 의존성(Dependency)과 하드웨어 자원이 허용하는 경우 독립적인 연산이 동시에 진행되도록 지원한다. 비동기 복사(Asynchronous Copy)와 이중 버퍼링(Double Buffering)은 현재 타일을 처리하는 동안 다음 데이터 타일(Data Tile)을 준비할 수 있다. 이러한 기법은 실행 유휴 시간(Idle Execution Time)을 줄이지만, 동기화는 데이터 의존성과 버퍼 재사용(Buffer Reuse)을 올바르게 보호해야 한다.

캐시 최적화(Cache Optimization)는 메모리 접근 패턴에서 시간적 지역성(Temporal Locality)과 공간적 지역성(Spatial Locality)을 활용하는 것을 목표로 한다. 반복적으로 접근하는 데이터는 L1 또는 L2 캐싱(Caching)의 이점을 얻을 수 있지만, 스트리밍 작업 부하(Streaming Workload)는 불필요한 캐시 압박(Cache Pressure)을 발생시킬 수 있다. 데이터 블로킹(Data Blocking)과 접근 순서 재구성(Access Reordering)은 사용 가능한 캐시 용량 내에서 데이터 재사용을 향상시킨다. 지원되는 아키텍처에서는 캐시 정책(Cache Policy)이 추가적인 제어 기능을 제공할 수 있지만, 그 효과는 프로파일링을 통해 검증해야 한다.

산술 집약도(Arithmetic Intensity)는 성능 한계(Performance Limit)를 이해하기 위한 유용한 분석 기준을 제공한다. 산술 집약도는 메모리에서 전송된 바이트 수(Bytes Transferred)에 대한 연산량(Computational Work)을 측정한다. 데이터 재사용을 증가시키면 산술 집약도가 높아지고 커널의 실행 특성이 대역폭 제한(Bandwidth-Limited)에서 연산 제한(Compute-Limited)으로 이동할 수 있다. 루프라인 모델(Roofline Model)은 산술 집약도와 달성 가능한 처리량(Achievable Throughput)을 연결하여 메모리 최적화와 산술 최적화 중 어느 쪽이 더 큰 개선 가능성을 제공하는지 판단하도록 돕는다.

병렬 리덕션(Parallel Reduction)은 여러 스레드가 공유 결과(Shared Result)에 기여하므로 특별한 주의가 필요하다. 효율적인 리덕션은 최종 출력을 생성하기 전에 워프 또는 블록 내부에서 부분 값(Partial Value)을 결합한다. 워프 셔플 연산(Warp Shuffle Operation)은 공유 메모리 트래픽을 줄일 수 있으며, 계층적 집계(Hierarchical Aggregation)는 동기화와 원자적 연산 경합(Atomic Contention)을 제한한다. 최적의 전략은 입력 크기(Input Size), 리덕션 차원(Reduction Dimension), 수치 정밀도 요구사항(Numerical Precision Requirement)에 따라 달라진다.

원자적 연산(Atomic Operation)은 공유 메모리 위치(Shared Memory Location)에 대한 올바른 갱신을 보장하지만, 많은 스레드가 동일한 주소에 접근하면 실행이 직렬화(Serialization)될 수 있다. 히스토그램(Histogram), 카운터(Counter), 희소 누산 커널(Sparse Accumulation Kernel)은 상당한 경합(Contention)을 경험할 수 있다. 레지스터나 공유 메모리 내부에서 중간 결과(Intermediate Result)를 개별적으로 유지하는 사유화(Privatization)는 충돌하는 갱신을 줄인다. 지역 결과(Local Result)를 계층적으로 결합하면 필요한 정확성을 유지하면서 처리량을 향상시킬 수 있다.

부하 균형(Workload Balancing)은 스레드가 불규칙한 양의 데이터를 처리할 때 중요해진다. 그래프 알고리즘(Graph Algorithm), 희소 연산(Sparse Computation), 가변 길이 시퀀스(Variable-Length Sequence)는 일부 워프가 유휴 상태(Idle State)에 있는 동안 다른 워프가 계속 실행되는 상황을 발생시킬 수 있다. 동적 작업 할당(Dynamic Work Assignment), 지속 실행 커널(Persistent Kernel), 압축된 작업 큐(Compacted Task Queue)는 활용률을 향상시킬 수 있다. 이러한 기법은 관리 오버헤드(Management Overhead)를 발생시키므로 효과는 작업 부하의 변동성(Workload Variability)과 작업 단위 크기(Task Granularity)에 따라 달라진다.

혼합 정밀도 최적화(Mixed-Precision Optimization)는 적절한 경우 낮은 정밀도의 수치 형식(Lower-Precision Numerical Format)을 사용하여 메모리 요구량을 줄이고 산술 처리량(Arithmetic Throughput)을 증가시킨다. 딥러닝 작업 부하는 일반적으로 저정밀도 곱셈(Reduced-Precision Multiplication)과 고정밀도 누산(Higher-Precision Accumulation)을 결합한다. 손실 스케일링(Loss Scaling), 수치 검증(Numerical Validation), 선택적 정밀도 유지(Selective Precision Retention)는 학습 안정성(Training Stability)을 보존하는 데 도움이 된다. 정밀도 선택(Precision Choice)은 하드웨어 처리량만이 아니라 모델 정확도 요구사항(Model Accuracy Requirement)을 반영해야 한다.

메모리 할당(Memory Allocation)과 전송 오버헤드(Transfer Overhead)는 작은 연산이 많은 애플리케이션의 성능에 상당한 영향을 미칠 수 있다. 장치 버퍼(Device Buffer)를 재사용하면 할당 비용(Allocation Cost)이 감소하며, 페이지 고정 호스트 메모리(Pinned Host Memory)는 효율적인 비동기 전송(Asynchronous Transfer)을 지원할 수 있다. CUDA 그래프(CUDA Graph)는 적합한 실행 시퀀스(Execution Sequence)에서 반복적인 커널 실행 오버헤드를 줄인다. 안정적인 메모리 수명(Memory Lifetime)과 예측 가능한 연산 의존성(Operation Dependency)은 효율적인 스케줄링과 실행 기회를 향상시킨다.

아키텍처 인식 최적화(Architecture-Aware Optimization)는 볼타(Volta), 암페어(Ampere), 에이다(Ada), 호퍼(Hopper) GPU 사이의 차이를 고려한다. 이러한 아키텍처는 텐서 코어 기능(Tensor Core Capability), 캐시 구성(Cache Organization), 비동기 실행 기능(Asynchronous Execution Feature), 자원 제한(Resource Limit)에서 차이가 있다. 한 세대에 최적화된 커널이 다른 세대에서 동일한 효율을 달성하지 못할 수 있다. 따라서 성능 튜닝(Performance Tuning)에서는 컴파일러 설정(Compiler Setting), 라이브러리 버전(Library Version), 지원되는 명령어 집합(Instruction Set)을 고려해야 한다.

프로파일링(Profiling)은 측정(Measurement), 가설 수립(Hypothesis Formation), 수정(Modification), 검증(Validation)의 반복적인 과정을 따라야 한다. 실제 달성 점유율(Achieved Occupancy), 메모리 처리량(Memory Throughput), 실행 정지(Execution Stall), 텐서 코어 활용률(Tensor Core Utilization)과 같은 지표는 서로 다른 성능 제한을 보여준다. 통제된 실험(Controlled Experiment)은 개별 변경사항의 효과를 분리하여 평가하는 데 도움이 된다. 벤치마킹(Benchmarking)에는 대표적인 입력 크기(Representative Input Size)와 적절한 동기화를 포함하여 비동기 실행으로 인한 잘못된 결론을 방지해야 한다.

딥러닝 최적화(Deep Learning Optimization)는 여러 연산 계층(Computational Layer)에 걸쳐 다양한 전략을 결합하는 경우가 많다. 행렬 곱셈은 타일링(Tiling)과 텐서 코어의 이점을 얻으며, 어텐션 커널(Attention Kernel)은 융합(Fusion)과 메모리 효율적인 데이터 재사용(Memory-Efficient Data Reuse)을 활용한다. 합성곱 성능(Convolution Performance)은 텐서 배치(Tensor Layout), 산술 집약도, 라이브러리 알고리즘 선택(Library Algorithm Selection)에 따라 달라진다. 효과적인 시스템 최적화(System Optimization)는 각 연산의 성능을 독립적으로 극대화하기보다 커널 사이의 상호작용(Interaction)을 고려한다.

로보틱스(Robotics)와 실시간 인공지능 작업 부하(Real-Time AI Workload)는 평균 처리량(Average Throughput)을 넘어서는 최적화를 요구한다. 인식 파이프라인(Perception Pipeline)은 센서 처리(Sensor Processing) 및 제어 시스템(Control System)과 자원을 공유하면서 지연시간 예산(Latency Budget)을 충족해야 한다. GPU 활용률(GPU Utilization), 메모리 전송, 실행 변동성(Execution Variability)은 종단 간 응답성(End-to-End Responsiveness)에 영향을 미친다. 따라서 최적화에서는 기존 성능 지표와 함께 최악 조건 동작(Worst-Case Behavior), 열 제한(Thermal Limit), 스케줄링 예측 가능성(Scheduling Predictability)을 평가해야 한다.

GPU 최적화는 궁극적으로 알고리즘(Algorithm), 수치 형식(Numerical Format), 메모리 계층 구조(Memory Hierarchy), 하드웨어 실행(Hardware Execution) 전반에 걸친 조정된 의사결정(Coordinated Decision)을 요구한다. 메모리 접근 병합(Coalescing), 타일링(Tiling), 커널 융합(Fusion), 비동기 실행(Asynchronous Execution), 점유율 튜닝(Occupancy Tuning)과 같은 기법은 서로 다른 병목을 해결한다. 어떤 단일 전략도 모든 작업 부하에서 최적 성능을 보장하지 않는다. 프로파일링 기반 실험(Profiling-Driven Experimentation)과 아키텍처 인식 설계(Architecture-Aware Design)는 확장 가능하고 효율적이며 신뢰성 높은 CUDA 애플리케이션(Scalable, Efficient, and Reliable CUDA Application)을 구축하기 위한 기반을 제공한다.

## 01.09 Applications

![](images/image10.png){width="7.268055555555556in" height="7.268055555555556in"}

GPU 아키텍처(GPU Architecture)는 대규모 병렬성(Massive Parallelism), 높은 메모리 대역폭(High Memory Bandwidth), 특수 연산 유닛(Specialized Computational Unit)을 결합하여 광범위한 애플리케이션(Application)을 지원한다. 딥러닝(Deep Learning), 시뮬레이션(Simulation), 로보틱스(Robotics)는 이러한 기능의 이점을 활용하는 세 가지 중요한 분야이다. 각 분야의 연산 요구사항(Computational Requirement)은 서로 다르지만, 모두 효율적인 실행(Efficient Execution), 메모리 관리(Memory Management), 확장 가능한 처리(Scalable Processing)에 의존한다. 이러한 응용 분야를 이해하면 GPU 하드웨어 아키텍처와 실제 시스템 설계(Practical System Design) 사이의 관계를 파악할 수 있다.

딥러닝은 GPU의 병렬 실행 자원(Parallel Execution Resource)에 효율적으로 매핑되는 행렬 곱셈(Matrix Multiplication), 합성곱(Convolution), 텐서 연산(Tensor Operation)에 크게 의존한다. 스트리밍 멀티프로세서(Streaming Multiprocessor, SM)는 수천 개의 스레드(Thread)를 조정하며, 텐서 코어(Tensor Core)는 지원되는 행렬 연산(Matrix Computation)을 가속한다. 고대역폭 메모리(High-Bandwidth Memory)는 모델 매개변수(Model Parameter)와 활성화 값(Activation)을 실행 유닛(Execution Unit)에 공급한다. 이러한 구성 요소들은 함께 작동하여 점점 복잡해지는 신경망(Neural Network)의 효율적인 학습(Training)과 추론(Inference)을 가능하게 한다.

신경망 학습(Neural Network Training) 과정에서 GPU는 대규모 데이터셋(Dataset)에 대해 순전파(Forward Propagation), 역전파(Backward Propagation), 매개변수 갱신(Parameter Update)을 수행한다. 혼합 정밀도 연산(Mixed-Precision Computation)은 메모리 사용량(Memory Consumption)을 줄이고 호환 가능한 산술 연산(Arithmetic Operation)을 가속한다. 텐서 코어는 행렬 연산 집약적 계층(Matrix-Intensive Layer)에 높은 처리량(Throughput)을 제공하며, 최적화된 라이브러리(Optimized Library)는 데이터 배치(Data Layout)와 커널 실행(Kernel Execution)을 관리한다. 학습 효율(Training Efficiency)은 수치 안정성(Numerical Stability), 메모리 용량(Memory Capacity), 연산 자원 활용률(Computational Utilization) 사이의 균형에 따라 결정된다.

대규모 언어 모델(Large Language Model, LLM)은 모델 매개변수와 중간 활성화 값(Intermediate Activation)이 개별 GPU의 용량을 초과할 수 있으므로 높은 메모리 및 통신 요구사항(Memory and Communication Requirement)을 발생시킨다. 데이터 병렬성(Data Parallelism)은 학습 데이터를 분산하고, 텐서 병렬성(Tensor Parallelism)과 파이프라인 병렬성(Pipeline Parallelism)은 모델 연산을 여러 장치에 분할한다. 고속 인터커넥트(High-Speed Interconnect)와 통신 라이브러리(Communication Library)는 분산 실행(Distributed Execution)을 조정한다. 전체 확장성(Overall Scalability)은 통신 오버헤드(Communication Overhead), 작업 부하 분할(Workload Partitioning), 동기화 효율(Synchronization Efficiency)에 따라 달라진다.

딥러닝 추론(Deep Learning Inference)은 응답 지연시간(Response Latency), 처리량(Throughput), 효율적인 메모리 사용(Efficient Memory Usage)을 중요하게 고려한다. GPU는 분류(Classification), 탐지(Detection), 언어 생성(Language Generation), 다중모달 추론(Multimodal Reasoning)을 위해 학습된 신경망을 실행한다. 저정밀도 산술 연산(Reduced-Precision Arithmetic)과 최적화된 실행 엔진(Optimized Execution Engine)은 자원 소비(Resource Consumption)를 제어하면서 성능을 향상시킬 수 있다. 배치 크기(Batch Size), 모델 아키텍처(Model Architecture), 메모리 접근 패턴(Memory Access Pattern)은 애플리케이션이 연산, 대역폭, 실행 오버헤드 중 어떤 요소에 의해 제한되는지를 결정한다.

컴퓨터 비전(Computer Vision) 애플리케이션은 영상 분류(Image Classification), 객체 탐지(Object Detection), 의미론적 분할(Semantic Segmentation), 시각적 추적(Visual Tracking)에 GPU 가속(GPU Acceleration)을 활용한다. 합성곱 신경망(Convolutional Neural Network)과 비전 트랜스포머(Vision Transformer)는 병렬 텐서 연산(Parallel Tensor Operation)을 통해 대량의 영상 데이터를 처리한다. 실시간 애플리케이션(Real-Time Application)은 효율적인 전처리(Preprocessing), 추론(Inference), 후처리(Postprocessing) 파이프라인(Pipeline)을 필요로 한다. 커널 융합(Kernel Fusion), 비동기 전송(Asynchronous Transfer), 최적화된 메모리 배치(Optimized Memory Layout)는 종단 간 처리 지연시간(End-to-End Processing Latency)을 줄이는 데 도움이 된다.

시뮬레이션(Simulation)은 다수의 독립적이거나 부분적으로 결합된 연산(Independent or Partially Coupled Computation)을 포함하는 수학적 모델(Mathematical Model)을 GPU로 가속한다. 물리 시스템(Physical System)은 입자(Particle), 격자(Grid), 유한 요소(Finite Element), 미분방정식(Differential Equation)을 통해 표현할 수 있다. GPU 스레드는 이러한 모델을 동시에 계산하여 적합한 작업 부하의 실행시간(Execution Time)을 단축한다. 성능은 수치 정밀도(Numerical Precision), 데이터 의존성(Data Dependency), 메모리 지역성(Memory Locality), 충분한 병렬성을 확보할 수 있는 능력에 따라 달라진다.

전산유체역학(Computational Fluid Dynamics, CFD)은 넓은 공간 영역(Spatial Domain)에 걸쳐 이산화된 방정식(Discretized Equation)을 해결할 때 GPU 가속의 이점을 얻는다. 병렬 커널(Parallel Kernel)은 계산 격자(Computational Grid) 전반에서 속도(Velocity), 압력(Pressure), 온도(Temperature), 기타 물리량(Physical Quantity)을 갱신한다. 메모리 대역폭과 인접한 격자 요소(Neighboring Grid Element) 사이의 통신은 성능에 큰 영향을 미친다. 효율적인 솔버(Solver)는 구조화된 메모리 접근(Structured Memory Access), 반복 알고리즘(Iterative Algorithm), 수치 해법(Numerical Method)에 적합한 동기화 전략(Synchronization Strategy)을 결합한다.

입자 기반 시뮬레이션(Particle-Based Simulation)은 유체(Fluid), 입상 물질(Granular Material), 상호작용하는 객체(Interacting Object)를 개별 요소(Discrete Element)의 집합으로 표현한다. GPU는 입자의 위치(Position), 속도(Velocity), 힘(Force)을 동시에 갱신할 수 있으므로 대규모 입자 집합(Large Particle Population)을 효율적으로 처리한다. 이웃 탐색(Neighbor Search)과 충돌 계산(Collision Calculation)은 불규칙한 메모리 접근 패턴(Irregular Memory Access Pattern)을 발생시킨다. 공간 분할(Spatial Partitioning)과 효율적인 자료구조(Data Structure)는 연산 복잡도(Computational Complexity)를 줄이고 병렬 활용률(Parallel Utilization)을 향상시키는 데 도움이 된다.

유한 요소 시뮬레이션(Finite Element Simulation)과 구조 해석(Structural Analysis)은 변형(Deformation), 응력(Stress), 동적 거동(Dynamic Behavior)을 설명하는 수학적 시스템(Mathematical System)을 구성하고 해결하는 과정을 포함한다. GPU는 적합한 희소 선형대수 연산(Sparse Linear Algebra Operation)과 요소 수준 연산(Element-Level Computation)을 가속한다. 그러나 불규칙한 연결 구조(Irregular Connectivity)와 동기화 요구사항(Synchronization Requirement)은 효율을 제한할 수 있다. 성능 최적화는 희소 행렬 표현(Sparse Matrix Representation), 병렬 솔버(Parallel Solver), 신중한 연산 작업 분배(Computational Work Distribution)를 결합하는 경우가 많다.

과학 시뮬레이션(Scientific Simulation)은 일반적인 신경망 추론보다 높은 수치 정밀도를 요구하는 경우가 많다. FP64 산술 연산(FP64 Arithmetic)은 일부 애플리케이션에서 수치 안정성(Numerical Stability), 수렴성(Convergence), 재현성(Reproducibility)을 확보하는 데 필수적일 수 있다. GPU 아키텍처는 배정밀도 처리량(Double-Precision Throughput)과 메모리 기능(Memory Capability)에서 상당한 차이를 보인다. 따라서 하드웨어 선택에서는 텐서 코어 사양(Tensor Core Specification)에만 의존하지 않고 수치 요구사항(Numerical Requirement)과 측정된 솔버 성능(Measured Solver Performance)을 고려해야 한다.

GPU 가속 시뮬레이션(GPU-Accelerated Simulation)은 디지털 트윈(Digital Twin), 합성 데이터 생성(Synthetic Data Generation), 강화학습 환경(Reinforcement Learning Environment)도 지원한다. 여러 시뮬레이션 인스턴스(Simulation Instance)를 동시에 실행하여 다양한 관측값(Observation)과 학습 경험(Training Experience)을 생성할 수 있다. 병렬 실행(Parallel Execution)은 데이터 생성 처리량(Data Generation Throughput)을 높이고 물리적 실험(Physical Experiment)에 대한 의존도를 줄인다. 그러나 시뮬레이션 충실도(Simulation Fidelity), 수치 안정성, 시뮬레이션 환경과 실제 환경 사이의 차이(Simulation-to-Reality Gap)는 여전히 중요한 고려사항이다.

로보틱스(Robotics)는 GPU 연산을 인식(Perception), 위치 추정(Localization), 지도 작성(Mapping), 경로 계획(Planning), 지능형 제어(Intelligent Control)와 통합한다. 카메라(Camera), 깊이 센서(Depth Sensor), 라이다(LiDAR) 시스템은 운영상 정해진 시간 제한(Operational Deadline) 내에 처리해야 하는 대량의 데이터를 생성한다. GPU는 영상 처리(Image Processing), 신경망 추론(Neural Inference), 기하학적 변환(Geometric Transformation), 일부 최적화 알고리즘(Optimization Algorithm)을 가속한다. 효과적인 활용을 위해서는 연산 성능(Computational Performance)과 실시간 시스템 요구사항(Real-Time System Requirement)을 조정해야 한다.

로봇 인식 파이프라인(Robotic Perception Pipeline)은 일반적으로 객체 탐지(Object Detection), 분할(Segmentation), 깊이 추정(Depth Estimation), 장면 이해(Scene Understanding)를 결합한다. 신경망(Neural Network)은 센서 관측값(Sensor Observation)을 처리하여 객체(Object), 장애물(Obstacle), 주행 가능 영역(Navigable Region)을 식별한다. 텐서 코어 가속(Tensor Core Acceleration)은 지원되는 추론 작업 부하(Inference Workload)의 성능을 높이고, 효율적인 메모리 전송(Memory Transfer)은 센서 입력에서 의사결정까지의 지연시간(Sensor-to-Decision Latency)을 줄인다. 신뢰성 높은 동작(Reliable Operation)을 위해서는 예측 불확실성(Prediction Uncertainty), 환경 변화(Environmental Variation), 실패 조건(Failure Condition)도 평가해야 한다.

동시적 위치 추정 및 지도 작성(Simultaneous Localization and Mapping, SLAM)은 센서 측정값(Sensor Measurement)을 결합하여 로봇의 움직임(Robot Motion)을 추정하고 환경 표현(Environmental Representation)을 구축한다. GPU 가속은 특징 추출(Feature Extraction), 영상 정합(Image Matching), 포인트 클라우드 처리(Point-Cloud Processing), 일부 최적화 절차(Optimization Procedure)를 지원할 수 있다. 병렬 알고리즘(Parallel Algorithm)은 밀집 지도 작성(Dense Mapping)과 대규모 센서 데이터셋(Large Sensor Dataset)의 처리량을 향상시킨다. 그러나 순차적 의존성(Sequential Dependency)과 불규칙한 그래프 구조(Irregular Graph Structure)는 파이프라인의 일부 단계에서 가속 효과를 제한할 수 있다.

자율 주행 및 이동(Autonomous Navigation)은 GPU 가속 인식(Perception)과 예측(Prediction)을 활용하여 동적 환경(Dynamic Environment)에서 안전한 이동을 지원한다. 신경망은 주행 가능성(Traversability)을 추정하고 장애물을 인식하며 주변 객체의 움직임(Object Motion)을 예측할 수 있다. 경로 계획 알고리즘(Planning Algorithm)도 후보 궤적(Candidate Trajectory)을 병렬로 평가함으로써 이점을 얻을 수 있다. 그러나 안전 필수 모션 제어(Safety-Critical Motion Control)는 GPU 추론에만 의존하기보다 실행시간이 제한된 동작(Bounded Execution Behavior)과 독립적인 안전 보호장치(Independent Safeguard)를 필요로 한다.

로봇 조작(Robotic Manipulation)은 객체 인식(Object Recognition), 자세 추정(Pose Estimation), 파지 계획(Grasp Planning), 충돌 없는 움직임 생성(Collision-Free Motion Generation)을 포함한다. GPU 기반 신경망(GPU-Based Neural Network)은 시각적 자세 추정(Visual Pose Estimation)과 학습 기반 파지 예측(Learned Grasp Prediction)을 지원한다. 병렬 기하 연산(Parallel Geometric Computation)은 충돌 검사(Collision Checking)와 후보 평가(Candidate Evaluation)를 가속할 수 있다. 로봇 제어기(Robot Controller)와 통합하려면 정확한 좌표 변환(Coordinate Transformation), 예측 가능한 통신(Predictable Communication), 명령된 움직임(Commanded Movement)에 대한 적절한 검증(Validation)이 필요하다.

로보틱스를 위한 강화학습(Reinforcement Learning for Robotics)은 GPU 시뮬레이션(GPU Simulation)과 신경망 학습(Neural Network Training)을 결합함으로써 이점을 얻는다. 수천 개의 시뮬레이션 환경(Simulated Environment)이 동시에 경험 데이터(Experience Data)를 생성하는 동안 가속된 텐서 연산(Accelerated Tensor Operation)을 사용하여 정책(Policy)을 갱신할 수 있다. 이러한 접근 방식은 학습시간(Training Time)을 줄이고 복잡한 제어 행동(Control Behavior)의 탐색(Exploration)을 지원한다. 실제 물리 환경에 성공적으로 배포하려면 도메인 무작위화(Domain Randomization), 안전성 검증(Safety Validation), 실제 동역학(Real-World Dynamics)에 대한 적응(Adaptation)이 여전히 필요하다.

엣지 로보틱스(Edge Robotics)는 중앙 집중식 데이터센터 컴퓨팅(Centralized Data-Center Computing)보다 엄격한 전력(Power), 열(Thermal), 메모리 제약조건(Memory Constraint)을 가진다. 임베디드 GPU 플랫폼(Embedded GPU Platform)은 이동 로봇(Mobile Robot)에서 직접 인식 및 의사결정 모델(Perception and Decision Model)을 실행하여 네트워크 연결(Network Connectivity)에 대한 의존도를 줄인다. 모델 양자화(Model Quantization), 효율적인 추론 엔진(Efficient Inference Engine), 신중한 스케줄링(Scheduling)은 지연시간 예산(Latency Budget)을 충족하는 데 도움이 된다. 시스템 설계자는 정확도(Accuracy), 연산 용량(Computational Capacity), 배터리 소비(Battery Consumption), 환경 운영 조건(Environmental Operating Condition) 사이의 균형을 유지해야 한다.

다중 로봇 시스템(Multi-Robot System)은 GPU 응용 범위를 개별 자율 기계(Autonomous Machine)를 넘어 확장한다. 중앙 집중식 또는 분산 컴퓨팅 자원(Centralized or Distributed Computing Resource)은 로봇 군집(Fleet)의 관측값을 처리하고 지도 작성(Mapping)과 협력적 경로 계획(Coordinated Planning)을 지원할 수 있다. 엣지 GPU(Edge GPU)는 시간 민감도가 높은 지역 추론(Time-Sensitive Local Inference)을 처리하고, 서버(Server)는 연산 집약적인 학습(Computationally Intensive Training)과 전역 분석(Global Analysis)을 수행한다. 통신 대역폭(Communication Bandwidth), 동기화(Synchronization), 네트워크 신뢰성(Network Reliability)은 연산 책임(Computational Responsibility)의 분배 방식에 영향을 미친다.

시뮬레이션, 딥러닝, 로보틱스의 융합(Convergence)은 통합된 피지컬 인공지능 시스템(Physical AI System)을 구축할 수 있는 기회를 제공한다. 시뮬레이션은 학습 데이터(Training Data)를 생성하고, 딥러닝은 유용한 표현(Representation)을 추출하며, 로봇 플랫폼(Robotic Platform)은 학습된 기능(Learned Capability)을 물리적 환경(Physical Environment)에 적용한다. GPU는 이러한 단계 전반에서 연산 가속(Computational Acceleration)을 제공하여 반복적인 개발(Iterative Development)과 검증(Validation)을 가능하게 한다. 효과적인 통합을 위해서는 일관된 데이터 인터페이스(Data Interface), 재현 가능한 실험(Reproducible Experiment), 현실적인 성능 평가(Realistic Performance Evaluation)가 필요하다.

애플리케이션 성능(Application Performance)은 GPU 아키텍처를 작업 부하 특성(Workload Characteristic)에 얼마나 적절하게 맞추는지에 따라 달라진다. 딥러닝은 텐서 코어 처리량(Tensor Core Throughput)의 이점을 얻고, 시뮬레이션은 FP64 성능이나 메모리 대역폭(Memory Bandwidth)을 우선시할 수 있으며, 로보틱스는 지연시간(Latency)과 전력 효율(Power Efficiency)을 강조하는 경우가 많다. 볼타(Volta), 암페어(Ampere), 에이다(Ada), 호퍼(Hopper)는 이러한 기능을 서로 다른 조합으로 제공한다. 하드웨어를 선택하려면 수치 형식(Numerical Format), 메모리 용량(Memory Capacity), 인터커넥트(Interconnect), 소프트웨어 호환성(Software Compatibility), 배포 제약조건(Deployment Constraint)을 검토해야 한다.

대표적인 조건(Representative Condition)에서 GPU 애플리케이션을 평가하려면 프로파일링(Profiling)과 벤치마킹(Benchmarking)이 필요하다. 딥러닝 측정 항목에는 학습 처리량(Training Throughput), 추론 지연시간(Inference Latency), 메모리 사용량(Memory Consumption)이 포함된다. 시뮬레이션 벤치마크(Simulation Benchmark)는 솔버 속도(Solver Speed), 수치 정확도(Numerical Accuracy), 확장성(Scalability)을 평가한다. 로보틱스 평가(Robotics Evaluation)는 추가로 센서 처리 마감시간(Sensor Processing Deadline), 종단 간 응답성(End-to-End Responsiveness), 운영 신뢰성(Operational Reliability)을 고려한다. 이러한 측정값은 최적화와 하드웨어 선택을 위한 실질적인 근거를 제공한다.

딥러닝, 시뮬레이션, 로보틱스에서 GPU를 활용하는 사례는 하드웨어-소프트웨어 공동 설계(Hardware-Software Co-Design)의 중요성을 보여준다. 병렬 실행(Parallel Execution), 특수 산술 연산(Specialized Arithmetic), 메모리 계층 구조(Memory Hierarchy), 효율적인 통신(Efficient Communication)은 고급 지능형 시스템(Advanced Intelligent System)을 위한 연산 기반(Computational Foundation)을 제공한다. 이러한 기술의 실질적인 가치는 수치적 제약조건(Numerical Constraint), 시간적 제약조건(Timing Constraint), 운영 제약조건(Operational Constraint)을 준수하면서 하드웨어 자원을 효과적으로 활용하는 알고리즘에 달려 있다. 이러한 통합은 확장 가능한 인공지능 개발(Scalable AI Development)과 더욱 높은 능력을 갖춘 피지컬 시스템(Physical System)을 가능하게 한다.
