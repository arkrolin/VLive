"""Live2D VLA — text/reference-conditioned motion3.json generation.

V7.1 architecture: param-as-token + 4-way identity embedding + additive
decomposition conditioning (text/gesture + static moc3 rig + B1 exemplars)
+ stable DiT (QKNorm / RMSNorm / MLP-dec). See outputs/Live2D方案V7.1_*.md.
"""
