#!/usr/bin/env python3
"""Optional extension: dynamic objects (4D) -- track a moving target, fuse its local mesh, export a ROS 2 bag.

    source /opt/ros/jazzy/setup.bash          # only for the bag export
    python reconstruct4d.py --config config/extensions/dynamic4d_kitti0059.json --output outputs/kitti0059_t51

See extensions/dynamic4d/INPUTS.md for the inputs."""
from extensions.dynamic4d.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
