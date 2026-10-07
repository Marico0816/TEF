"""Optional extension: dynamic objects (4D).  Off and not imported in the paper configuration.

Tracks one moving target through a case interval on the given sensor poses, refines its 4-DoF pose with image features,
fuses its returns into an object-frame TSDF, extracts the object mesh, and exports estimates, meshes and images as a
ROS 2 (MCAP) bag for RViz.  Entry point: ``reconstruct4d.py``; inputs and their formats: ``INPUTS.md``.
"""
