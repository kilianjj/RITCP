"""
Main program for creating and running the incremental SfM pipeline.
Create an IncrementalSFMEngine instance and register desired image streams.
"""

from incremental_sfm.incremental_sfm_engine import IncrementalSFMEngine
from utils.image_stream import ImageStream
import os

if __name__ == "__main__":
    # data = "car_data"
    # proj = "test15"
    # data = "data"
    # proj= "batch_three"

    data = "eval_data"
    proj = "eval5"

    sfm = IncrementalSFMEngine(
        data_path=data,
        project_name=proj,
        config_path="sfm_config.yaml",
    )

    # eval set
    for i in range(1, 4):
        sfm.register_stream(
            ImageStream(
                os.path.join(data, "images", f"car{i}")
            )
        )

    # desk example
    # for i in range(1, 11):
    #     sfm.register_stream(
    #         ImageStream(
    #             os.path.join(data, "images", f"new{i}")
    #         )
    #     )

    # rushil dir model
    # for i in range(1, 17):
    #     sfm.register_stream(
    #         ImageStream(
    #             os.path.join(data, "images", f"vehicle_{i:02}")
    #         )
    #     )

    ####### example of multiple cars
    # for i in range(1, 11):
    #     # streams 4 and 8 don't localize with the images randomly selected so would need vocab tree again, ONLY 1,2,3,6 localize with 1000 features
    #     # good = [1, 2, 3, 6]
    #     # if not i in good:
    #     #     continue
    #     if i % 4 == 0: continue
    #     sfm.register_stream(
    #         ImageStream(
    #             os.path.join(data, "images", f"stream{i}")
    #         )
    #     )

    ####### example of 2 cars where they each send 5 frames at a time, so we can test batch processing
    # stream_path1 = os.path.join(data, "images", "new1")
    # stream_path2 = os.path.join(data, "images", "new2")
    # stream1 = ImageStream(stream_path1)
    # stream2 = ImageStream(stream_path2)
    # sfm.register_stream(stream1)
    # sfm.register_stream(stream2)

    sfm.incremental_sfm()