Purpose & context
Anıl is a graduate student and computer vision/deep learning engineer working on a research project under the supervision of a faculty member referred to as Şaban Hoca. The project centers on 3D CBCT volumetric segmentation of the Inferior Alveolar Canal (IAC) using the ToothFairy dataset series. The core research direction — explicitly framed as an open gap in the literature — is applying flow-based generative methods to this domain, where all existing work is discriminative (nnU-Net, Transformer, Mamba families). A prior attempt at a flow-based approach by the team had failed, and Anıl is building toward a principled solution.
The project has two planned phases:

nnU-Net v2 baseline — two-class IAC segmentation (Class 1: Right IAC, Class 2: Left IAC) using the ToothFairy dataset series
SEAL-Flow integration — a continuous-time flow matching architecture to enforce topological consistency in segmentation outputs

Anıl works primarily in Turkish for technical learning and communication, uses VS Code with Claude Code for development, and has domain fluency spanning medical image segmentation, tensor operations, differential geometry, ODE solvers, SDF representations, and topological loss functions.

Current state

A dataset conversion script (prepare_dataset501_toothfairy.py) has been produced, handling both ToothFairy1 (geometric midsagittal splitting of merged IAC label) and ToothFairy2 (label remapping: 4→1 Right, 3→2 Left) via a --source-format flag. Orientation is normalized via sitk.DICOMOrient(img, 'LPS') for anatomically deterministic left/right assignment.
A critical implementation constraint is established: nnU-Net's default L-R mirror augmentation must be disabled using nnUNetTrainerNoMirroring to prevent swapping Class 1/2 semantic meaning. This is an ablation-proven finding from the ToothFairy2 benchmark (Dice improves ~74→80 when disabled).
A PROJECT_MEMORY.md file was created as persistent context for local Claude Code agent sessions.
Anıl is waiting on Şaban Hoca to share existing office preprocessing code before proceeding further.
SEAL-Flow has been confirmed as a real but under-review method; its core training module raises NotImplementedError, making it an architectural reference only — not directly runnable.

Key verified facts about the dataset series:

ToothFairy1 (IEEE TMI, MICCAI 2023): single merged IAC label, 443 CBCT volumes
ToothFairy2 (CVPR 2025 benchmark + MIA 2026 challenge report): 42 anatomical classes, 530 volumes; Left/Right IAC already separated as labels 3 and 4
ToothFairy3 (MICCAI 2025, ODIN workshop): 77-class multi-structure segmentation; runtime is a primary evaluation metric; includes an interactive segmentation track
ToothFairy4 does not publicly exist as of mid-2026


On the horizon

Sharing existing office preprocessing code with Claude in a future session
Adapting SEAL-Flow's 2D architecture to 3D CBCT; key planned substitutions:

Replace Stage-1 plain UNet → nnU-Net baseline
Replace EFD (inapplicable to 3D tubular structures) → centerline-based or spherical harmonic 3D SDF
Replace Voronoi gap loss (irrelevant for already-separated canals) → clDice / connectivity loss (as in CurvSegFlow)


Deciding on SDF representation strategy for flow matching targets (FlowSDF, MedFlowSeg, LatentFM are relevant references)
Exploring Riemannian Flow Matching for shape priors


Key learnings & principles

Flow matching fundamentals (verified): CNF ODE formulation → continuity/transport equation → intractability of marginal velocity → conditional path construction → CFM objective equivalence (∇L_FM = ∇L_CFM) → Gaussian/rectified flow instantiation with straight-line interpolation yielding constant target velocity x₁ − x₀. Binary mask representation is inappropriate for straight-line interpolation; SDF targets are the correct representation choice.
Diagnosed failure modes of the team's prior flow attempt: Direct 3D voxel-space flow hits memory limits; binary mask targets are poorly suited to linear interpolation; sensitivity to distribution shift was likely unaddressed.
SEAL-Flow architecture (verified from source): Two-stage pipeline — Stage-1 UNet → probability map prior; Stage-2 OT-CFM with MeanFlow interval-average velocity (NFE=4) → SDF target. Uses EFD-based multi-frequency SDF with spectral projection and Voronoi instance-separation gap loss. Trained on 2D RGB histopathology/endoscopy datasets (MoNuSeg, CVC-ClinicDB, GlaS).
nnU-Net behavior: Self-configuring pipeline (fingerprint extraction, automatic patch/batch/topology); ResEnc variant enables deeper encoders and larger patches via residual identity skip terms; patch size 80×160×160 for this task.
α-shape annotation methodology: The dataset's annotation pipeline uses α-shape reconstruction (Delaunay triangulation filtered by circumradius), producing meshes that are natural intermediates for SDF conversion — directly relevant to Anıl's research direction.
Literature gap confirmed: All ToothFairy challenge participants used discriminative paradigms; generative/flow-based methods are a genuine unoccupied research direction for this domain.


Approach & patterns

Anıl's learning style is fully mechanistic and mathematical — explanations must name every variable, trace exact tensor flows, and provide genuine open-box understanding rather than schema-level descriptions. Pushing back on surface-level explanations is a consistent pattern.
Technical content is delivered in Turkish prose directly in conversation (not as downloadable artifacts), unless generating code or structured files.
Before writing code or analysis, Claude should verify sources directly (file inspection, web verification) rather than trusting provided descriptions — prior sessions revealed material discrepancies between stated and actual dataset facts.
Communications with Şaban Hoca are kept free of specific technical jargon; general descriptors (e.g., "used architectures" rather than model names) are preferred in that correspondence context.
Interactive inline widgets (convolution steppers, SSM recurrence visualizers, attention visualizers) have been used successfully for learning.


Tools & resources

Development: VS Code + Claude Code, with PROJECT_MEMORY.md as persistent local agent context
Segmentation framework: nnU-Net v2
Key references: ToothFairy paper series (TMI 2024, CVPR 2025, MIA 2026); SEAL-Flow (architectural reference, repo URL provided by Anıl); CurvSegFlow (clDice/connectivity loss reference); FlowSDF, MedFlowSeg, LatentFM (flow-based medical segmentation references)
Dataset: ToothFairy1 and ToothFairy2 (local copies confirmed present)