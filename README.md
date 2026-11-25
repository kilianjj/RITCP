# RITCP: Rapid Infrastructure Triangulation Cooperative Perception

An image-only cooperative perception framework. A central orchestration server device holds a static Structure-from-Motion map of the environment. Client devices send their image feeds to the central server which quickly triangulates new points from the new set of images. The resulting 3D point cloud is semantically segmented and detections are produced. Detections can be sent back to client devices to extend their perception, ideally improving safety and awareness.

This was my implementation branch of a school project I worked on for an autonomous driving course along with my friend Rushil before we merged our initial work fully. Here's our [report](./SfmProjectReportRITCP.pdf) if you're curious. Would be fun to continue working on this if I had time. 
