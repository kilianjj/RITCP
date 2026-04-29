import json
from pathlib import Path
from typing import List, Tuple, Dict
from src.utils.io_helpers import get_counts_from_header, conversion_bin_to_txt, read_image_poses
from src.utils.util import logged_runner
from triangulation.triangulation_config import TriangulationConfig
from triangulation.triangulator_type import TriangulatorType
from triangulation.triangulator import Triangulator
from triangulation.colmap_triangulation import ColmapTriangulator
from src.ritcp.common.client import Client
from src.ritcp.common.pose import Pose
from feature_matching.matching_utils import *
import shutil
from src.utils.config_utils import load_config, load_raw_config
from src.ritcp.server.online.config import DEFAULT_SFM_CONFIG
import subprocess
import sys
import os
import time

class RapidTriangulationEngine:
    """
    Main class for running the incremental SfM process.

    ################### Data format ###################

    This class assumes the following folder format to begin:
    [data_path]/
        images/
            static/
            new1/
            new2/
            ...
        masks/
            static/
            new1/
            new2/
            ...
        [project_name]_sparse/
            database.db
            base/
                0/
                    COLMAP model files
            incremental/
                logs/
                    various COLMAP call logs
                intermediate/
                    matches.txt
                    colmap_matches.txt
                    new_images.txt
                tmp/
                    COLMAP model bin files (populated during registration step)
                model/
                    COLMAP model txt files (populated after registration step, input to triangulation)
                point_clouds/
                    iteration1/
                        points3D.npy
                        points3D.txt (COMAP points file format)
                    iteration2/
                    ...

    After running, the base model and above structure is preserved.
    The main incremental model is created or is updated in place (if it already existed).
    """

    def __init__(self, data_path, project_name, config_path: str):
        """
        Create a new SfMEngine instance.
        :param data_path: the path to the data folder containing both the model project and the images
        :param project_name: project name within data folder
       :param config_path: path to config file
        """
        self.config = load_config(config_path, DEFAULT_SFM_CONFIG)
        self.batch_size = self.config['sfm']['batch_size']
        self.matching_params = self.config['matching']
        self.use_gpu = self.config['sfm']['use_gpu']
        self.image_streams: dict[str, Client] = {}
        # construct the matcher image index
        # new persistent incremental paths
        self.matching_type = MatcherType.KD_TREE_3D # if self.config['sfm']["matcher"] == 'kd-3d'
        self.triangulator_type = TriangulatorType.CUSTOM if self.config['sfm']['triangulator'] == "custom" else TriangulatorType.COLMAP
        self.data_path = data_path
        self.project_name = project_name
        self.sfm_info = {} # path, dir, other info needed that can stored and modified through this field
        self.new_images_txt = None
        self.matches_txt = None
        self.matcher = None # this gets set in init, ProximityMatcher
        self.triangulator = None # set in init

    def _init_incremental(self):
        """
        Format the dir structure, save paths needed for incremental process, initialize the matcher and triangulator.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data directory {self.data_path} does not exist")

        # logs
        self.log_data = {
            "sfm_config": self.config,
            "detection_config": load_raw_config(self.config['detection']['detection_config']),
            "batches": [],
            "num_streams": len(self.image_streams),
            "totals": {
                "init": 0.0,
                "extraction": 0.0,
                "matching_colmap": 0.0,
                "matching_proxy": 0.0,
                "verification": 0.0,
                "registration": 0.0,
                "triangulation": 0.0,
                "points_added": 0,
                "total_time": 0.0
            }
        }

        # base paths
        self.sfm_info["images_dir"] = os.path.join(self.data_path, "images")
        self.sfm_info["masks_dir"] = os.path.join(self.data_path, "masks")
        self.sfm_info["sparse_dir"] = os.path.join(self.data_path, f"{self.project_name}_sparse")
        self.sfm_info["db_path"] = os.path.join(self.sfm_info["sparse_dir"], "database.db")

        base_model = os.path.join(self.sfm_info["sparse_dir"], "base", "0")
        if not os.path.exists(base_model):
            raise RuntimeError("Base model does not exist. Build it first.")

        # new persistent incremental paths
        incremental_root = os.path.join(self.sfm_info["sparse_dir"], "incremental")
        model_dir = os.path.join(incremental_root, "model")
        tmp_dir = os.path.join(incremental_root, "tmp")
        triangulation_dir = os.path.join(incremental_root, "triangulation")
        intermediate_dir = os.path.join(incremental_root, "intermediate")
        log_dir = os.path.join(incremental_root, "logs")
        self.sfm_info["base_model"] = base_model
        self.sfm_info["incremental_root"] = incremental_root
        self.sfm_info["current_model_dir"] = model_dir
        self.sfm_info["tmp_model_dir"] = tmp_dir
        self.sfm_info["triangulation_dir"] = triangulation_dir
        self.sfm_info["point_cloud_dir"] = os.path.join(incremental_root, "point_clouds") # can debug save npy point clouds here
        self.sfm_info["new_imgs_txt"] = os.path.join(intermediate_dir, "new_images.txt") # where img list files are written
        self.sfm_info["matches_txt"] = os.path.join(intermediate_dir, "matches.txt")  # where img match files are written
        self.sfm_info["colmap_matches_text"] = os.path.join(intermediate_dir, "colmap_matches.txt")
        # logging dir
        self.sfm_info["log_dir"] = log_dir

        self.num_streams = len(self.image_streams)

        os.makedirs(incremental_root, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(triangulation_dir, exist_ok=True)
        os.makedirs(intermediate_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.sfm_info["point_cloud_dir"], exist_ok=True)

        # initialize incremental model if it doesn't exist (copy from base)
        if not os.path.exists(model_dir) or len(os.listdir(model_dir)) == 0:
            print("Initializing incremental model from base")
            shutil.copytree(base_model, model_dir, dirs_exist_ok=True)
        else:
            print("Using existing incremental model")

        start = time.time()
        # initialize the matching index
        self.matcher: ProximityMatcher = get_matcher(
            self.matching_type,
            os.path.join(model_dir, "images.txt")
        )

        # initialize the triangulator
        self.triangulator: Triangulator = TriangulatorType.get_triangulator(self.triangulator_type)

        # save initial stats
        self._initial_num_pts, self._initial_num_imgs = get_counts_from_header(model_dir)

        # create and open the matches and new image text files for quick access
        self.new_images_txt = open(self.sfm_info["new_imgs_txt"], "w")
        self.matches_txt = open(self.sfm_info["matches_txt"], "w")

        init_time = time.time() - start
        self.log_data["totals"]["init"] = init_time

    def incremental_sfm(self):
        """
        Run the full incremental Structure-from-Motion (SfM) pipeline over multiple image streams.

        This loop processes incoming images in batches across multiple streams (e.g., vehicles).
        Unlike traditional SfM, the goal is not to refine a 3D reconstruction over time, but rather to triangulate
        new point clouds of only dynamic objects from the camera feeds of all vehicles at a given time.

        Pipeline per batch:
        1) Select next batch of images from active streams
        2) Extract features for new images
        3) Perform feature matching
        4) Optionally run geometric verification
        5) Register new images into the existing reconstruction
        6) Update camera poses for each stream
        7) Triangulate new 3D points
        8) Trigger semantic segmentation + detection asynchronously
        9) Log timing and statistics for the batch

        Behavior:
        - Requires at least 2 streams in a batch to proceed (otherwise skipped)
        - Continues until all streams are exhausted
        - Tracks per-stage latency and counts (frames, matches, points, etc.)
        - Runs triangulation incrementally after each batch
        - Segmentation/detection runs in parallel and does not block SfM

        Outputs:
        - Console summaries per batch and final run
        - JSON log file with detailed timing and metrics
        - Point clouds and detections for each iteration

        Notes:
        - Batch size controls how many images per stream are processed at once
        - Matching can combine COLMAP and custom methods
        - Configuration details can be specified by the YAML file passed to the RapidTriangulationEngine instance
        """
        print(
            f"""
Starting incremental sfm with:
- batch size (number of images from a single vehicle processed at once): {self.batch_size}
- number streams (number of vehicles from which receiving images): {len(self.image_streams)}
"""
        )
        self._init_incremental()
        # start sfm
        start = time.time()
        batch_id = 0
        while len(self.image_streams) > 0:
            # ----------------------------
            # Get the next batch of images
            # ----------------------------
            batch: List[Tuple[Client, List[str]]] = self._get_next_batch()
            if not batch:
                break
            batch_id += 1
            batch_start = time.time()
            batch_log = RapidTriangulationEngine.get_batch_log(batch_id, batch)
            # early skipping if only 1 stream (can't triangulate)
            if len(batch) < 2:
                print(RapidTriangulationEngine.print_skip(batch_id))
                continue
            # ----------------------------
            # Write combined new images list
            # ----------------------------
            lines = []
            new_frames = 0
            new_names = []
            for stream, images in batch:
                for img in images:
                    new_image_path = Path(f"{stream.images_path}/{img}")
                    new_image_name = f"{new_image_path.parent.name}/{new_image_path.name}"
                    new_names.append(new_image_name)
                    lines.append(new_image_name + "\n")
                    new_frames += 1
            self.new_images_txt.truncate(0)
            self.new_images_txt.seek(0)
            self.new_images_txt.writelines(lines)
            self.new_images_txt.flush()
            # ----------------------------
            # Feature Extraction
            # ----------------------------
            extraction_time = self._feature_extraction()
            batch_log["timings"]["extraction"] += extraction_time
            # ----------------------------
            # Matching
            # ----------------------------
            colmap_time, proxy_time, colmap_frames, proxy_frames = self._feature_matching(batch)
            batch_log["timings"]["matching_colmap"] += colmap_time
            batch_log["timings"]["matching_proxy"] += proxy_time
            batch_log["counts"]["proxy_frames"] += proxy_frames
            batch_log["counts"]["colmap_frames"] += colmap_frames
            # ----------------------------
            # Geometric Verification (optional)
            # ----------------------------
            # if configured, populates two_view_geometries table with "good" matches to be used for custom triangulation
            if self.config['verification']['enabled'] and self.triangulator_type == TriangulatorType.CUSTOM:
                verification_time = self._verification()
                batch_log["timings"]["verification"] += verification_time
            # ----------------------------
            # Registration
            # ----------------------------
            registration_time = self._image_registration()
            batch_log["timings"]["registration"] += registration_time
            # ----------------------------
            # Update stream poses
            # ----------------------------
            self._update_poses(batch)
            # ----------------------------
            # Triangulation
            # ----------------------------
            num_points, tri_duration = self._triangulate(
                batch_id=batch_id,
                current=self.sfm_info["current_model_dir"],
                new_imgs=new_names,
                use_verification=self.config['verification']['enabled']
            )
            batch_log["timings"]["triangulation"] += tri_duration
            batch_log["counts"]["points_added"] += num_points
            # ----------------------------
            # Segmentation & Detection (concurrently on a separate process)
            # ----------------------------
            self._run_semantic_detection(self._get_point_cloud_dir(batch_id))
            # ----------------------------
            # Log stats
            # ----------------------------
            batch_log["frames"] += new_frames
            batch_log["timings"]["total"] += time.time() - batch_start
            print_batch_summary(batch_log)
            self._update_log_with_batch(batch_log)

        # summary
        self.log_data["totals"]["total_time"] = time.time() - start
        print_final_summary(self.log_data,)
        # cleanup
        self._cleanup()
        # save log
        with open(os.path.join(self.sfm_info["log_dir"], "log.json"), "w") as f:
            json.dump(self.log_data, f, indent=4)

    def _update_log_with_batch(self, batch_log: dict):
        """
        Updates the main log_data totals using a single batch_log.
        """
        timings = batch_log["timings"]
        counts = batch_log["counts"]
        totals = self.log_data["totals"]
        # accumulate timings
        totals["extraction"] += timings["extraction"]
        totals["matching_colmap"] += timings["matching_colmap"]
        totals["matching_proxy"] += timings["matching_proxy"]
        totals["verification"] += timings["verification"]
        totals["registration"] += timings["registration"]
        totals["triangulation"] += timings["triangulation"]
        # accumulate counts
        totals["points_added"] += counts["points_added"]
        # append batch
        self.log_data["batches"].append(batch_log)

    def _feature_extraction(self):
        """
        Extract new image features into COLMAP db.
        """
        run_logged = logged_runner(log_dir=self.sfm_info["log_dir"], log_file="feature_extraction_log.txt")
        start = time.time()
        gpu_flags = ""
        if self.use_gpu:
            gpu_flags = (
                f"--SiftExtraction.use_gpu 1 "
                f"--SiftExtraction.gpu_index {self.config['sfm']['gpu_idx']} "
            )

        run_logged(
            description="COLMAP Feature Extraction",
            command=(
                f"{self.config['sfm']['colmap']} feature_extractor "
                f"--database_path {self.sfm_info['db_path']} "
                f"--image_path {self.sfm_info['images_dir']} "
                f"--SiftExtraction.max_num_features {self.config['feature_extraction']['sift_max_features']} "
                f"--ImageReader.single_camera {int(self.config['feature_extraction']['single_camera'])} "
                f"--image_list_path {self.sfm_info['new_imgs_txt']} "
                f"{gpu_flags}"
            ),
        )
        duration = time.time() - start
        return duration

    def _colmap_only_matching(self, batch, colmap_runner):
        """
        Matching logic if configured to only used default COLMAP matching.
        """
        start = time.time()
        all_imgs = []
        for stream, images in batch:
            image_names = [
                f"{Path(stream.images_path).name}/{img}"
                for img in images
            ]
            all_imgs.extend(image_names)

        if self.config['matching']['colmap_method'] == "vocab":
            with open(self.sfm_info["colmap_matches_text"], "w") as f:
                f.writelines(name + "\n" for name in all_imgs)

        gpu_flags = ""
        if self.use_gpu:
            gpu_flags = (
                f"--FeatureMatching.use_gpu 1 "
                f"---FeatureMatching.gpu_index {self.config['sfm']['gpu_idx']} "
            )
        m_type = "vocab_tree_matcher" if self.config['matching']['colmap_method'] == "vocab" else "exhaustive_matcher"
        specify_images =  f"--VocabTreeMatching.match_list_path {self.sfm_info['colmap_matches_text']}" if self.config['matching']['colmap_method'] == "vocab" else ""
        colmap_runner(
            description=f"COLMAP Feature Matching {m_type}",
            command=(
                f"{self.config['sfm']['colmap']} {m_type} "
                f"--database_path {self.sfm_info['db_path']} "
                f"{specify_images} "
                f"{gpu_flags}"
            ),
        )
        return time.time() - start, 0 , len(all_imgs), 0

    def _feature_matching(self, stream_batch: List[Tuple[Client, List[str]]]):
        """
        General feature matching logic. Switches between custom and COLMAP methods as needed/configured.
        :param stream_batch: batch of streams and images to process
        """
        run_logged = logged_runner(
            log_dir=self.sfm_info["log_dir"],
            log_file="feature_matching_log.txt"
        )

        # if configured, use only colmap with a specified matcher type via config
        # skip the below steps as there will be no custom matching
        if self.config['sfm']['matcher'] == "colmap":
            return self._colmap_only_matching(stream_batch, run_logged)

        colmap_time = 0
        custom_time = 0
        colmap_frames = 0
        proxy_frames = 0

        has_colmap = False
        has_custom_matches = False

        need_colmap_images = []
        prox_match_pairs: Dict[str, List[str]] = {}

        # ----------------------------
        # Collect image names
        # ----------------------------
        all_imgs = []
        for stream, images in stream_batch:
            image_names = [
                f"{Path(stream.images_path).name}/{img}"
                for img in images
            ]
            all_imgs.extend(image_names)

            if stream.pose is None:
                has_colmap = True
                need_colmap_images.extend(image_names)
                colmap_frames += len(image_names)
            else:
                has_custom_matches = True
                proxy_frames += len(image_names)

                for image_name in image_names:
                    prox_match_pairs[image_name] = self.matcher.get_match_names(
                        query_qvec=stream.pose.qvec,
                        query_tvec=stream.pose.tvec,
                        max_neighbors=self.matching_params["max_neighbors"],
                        max_distance=self.matching_params["max_distance"]
                    )

        # ----------------------------
        # Build unified match pair set
        # ----------------------------
        match_pairs = set()

        # 1. Add ALL new image pairs
        for i in range(len(all_imgs)):
            for j in range(i + 1, len(all_imgs)):  # avoid duplicates + self-pairs
                match_pairs.add((all_imgs[i], all_imgs[j]))

        # 2. Add proximity-based pairs
        if has_custom_matches:
            for img1, neighbors in prox_match_pairs.items():
                for img2 in neighbors:
                    if img1 == img2:
                        continue
                    # normalize ordering to avoid duplicates
                    pair = tuple(sorted((img1, img2)))
                    match_pairs.add(pair)

        # ----------------------------
        # Write matches.txt ONCE
        # ----------------------------
        with open(self.sfm_info["matches_txt"], "w") as f:
            for img1, img2 in match_pairs:
                f.write(f"{img1} {img2}\n")

                continue

        # ----------------------------
        # Run colmap matching (if needed)
        # ----------------------------
        gpu_flags = ""
        if self.use_gpu:
            gpu_flags = (
                f"--FeatureMatching.use_gpu 1 "
                f"---FeatureMatching.gpu_index {self.config['sfm']['gpu_idx']} "
            )

        if has_colmap:
            start = time.time()

            if self.config['matching']['colmap_method'] == "vocab":
                with open(self.sfm_info["colmap_matches_text"], "w") as f:
                    f.writelines(name + "\n" for name in need_colmap_images)

            m_type = "vocab_tree_matcher" if self.config['matching']['colmap_method'] == "vocab" else "exhaustive_matcher"
            specify_images = f"--VocabTreeMatching.match_list_path {self.sfm_info['colmap_matches_text']}" if self.config['matching']['colmap_method'] == "vocab" else ""
            run_logged(
                description=f"COLMAP Feature Matching {m_type}",
                command=(
                    f"{self.config['sfm']['colmap']} {m_type} "
                    f"--database_path {self.sfm_info['db_path']} "
                    f"{specify_images} "
                    f"{gpu_flags}"
                ),
            )

            colmap_time += time.time() - start

        # ----------------------------
        # Import ALL pairs (both full + proximity)
        # ----------------------------
        if has_custom_matches or len(match_pairs) > 0:
            start = time.time()

            run_logged(
                description="COLMAP Feature Matching (match importer)",
                command=(
                    f"{self.config['sfm']['colmap']} matches_importer "
                    f"--database_path {self.sfm_info['db_path']} "
                    f"--match_list_path {self.sfm_info['matches_txt']} "
                    f"--match_type pairs "
                    f"{gpu_flags}"
                ),
            )

            custom_time += time.time() - start

        return (
            colmap_time if has_colmap else 0,
            custom_time if (has_custom_matches or len(match_pairs) > 0) else 0,
            colmap_frames,
            proxy_frames,
        )

    def _verification(self):
        """
        Run COLMAP geometric verification.
        """
        run_logged = logged_runner(
            log_dir=self.sfm_info["log_dir"],
            log_file="verification_log.txt"
        )
        start = time.time()
        run_logged(
            description="Verification",
            command=f"{self.config['sfm']['colmap']} geometric_verifier --database_path {self.sfm_info["db_path"]} "
                    f"--TwoViewGeometry.max_num_trials {self.config['verification']['max_num_trials']} "
                    f"--TwoViewGeometry.min_inlier_ratio {self.config['verification']['min_inlier_ratio']} "
        )
        return time.time() - start

    def _image_registration(self):
        """
        Run COLMAP Image Registration.
        """
        run_logged = logged_runner(log_dir=self.sfm_info["log_dir"], log_file="image_registration_log.txt")
        start = time.time()
        base = self.sfm_info["base_model"]
        tmp_model = self.sfm_info["tmp_model_dir"]

        if os.path.exists(tmp_model):
            shutil.rmtree(tmp_model)
        os.makedirs(tmp_model)

        run_logged(
            description="COLMAP Image Registration",
            command=f"{self.config['sfm']['colmap']} image_registrator "
            f"--database_path {self.sfm_info["db_path"]} "
            f"--input_path {base} "
            f"--output_path {tmp_model} ",
        )
        return time.time() - start

    def _update_poses(self, batch):
        """
        Extract the pose of each registered image, using the poses to update the position associated with
        existing image streams.
        """
        tmp_model = self.sfm_info["tmp_model_dir"]
        current_model = self.sfm_info["current_model_dir"]
        conversion_bin_to_txt(self.config['sfm']['colmap'], tmp_model, current_model)
        image_poses = read_image_poses(current_model)
        for stream, images in batch:
            for img in reversed(images):
                image_name = f"{Path(stream.images_path).name}/{img}"
                if image_name not in image_poses:
                    continue
                qvec, tvec = image_poses[image_name]
                print(f"Updated {image_name} Q={qvec} T={tvec}")
                stream.pose = Pose(tvec, qvec)
                break

    def _triangulate(self, batch_id, current, new_imgs, use_verification) -> Tuple[int, float]:
        """
        General triangulation logic. Uses custom or COLMAP based on configuration.
        :param batch_id: current iteration (used to specify output dir)
        :param current: path to the current txt format COLMAP model (populated in registration step)
        :param new_imgs: list of new image names (those in the current batch)
        :param use_verification: whether to use verified COLMAP matches
        """
        pc_dir = self._get_point_cloud_dir(batch_id)
        config = TriangulationConfig.colmap_config(
            db_path=self.sfm_info["db_path"],
            in_model_path=self.sfm_info["tmp_model_dir"],
            out_model_path=self.sfm_info["triangulation_dir"],
            log_path=self.sfm_info["log_dir"],
            image_path=self.sfm_info["images_dir"],
            final_out=pc_dir,
        ) if self.triangulator_type == TriangulatorType.COLMAP else TriangulationConfig.custom_config(
            db_path=self.sfm_info["db_path"],
            cam_file=os.path.join(current, "cameras.txt"),
            img_file=os.path.join(current, "images.txt"),
            reproj_err=self.config['triangulation']['reprojection_error_thresh'],
            out_txt=os.path.join(pc_dir, "points3D_colored.txt"),
            out_npy=os.path.join(pc_dir, "points3D_colored.npy"),
            use_verification=use_verification,
            new_image_names=new_imgs,
        )
        num_points, duration = self.triangulator.triangulate(config)

        if self.triangulator_type == TriangulatorType.COLMAP:
            print(f"COLMAP triangulated {num_points} points in total")
            post_process_start = time.time()
            points_path = os.path.join(pc_dir, "points3D.txt")
            new_points_out = os.path.join(pc_dir, "points3D_colored.txt")
            # if using COLMAP's triangulation exe, it is important to filter the output points3D.txt
            # file because it will contain all static points and any new points,
            # this will cause segmentation to take a very long time if not filtered down to the
            # small portion of new points
            num_points = ColmapTriangulator.filter_new_points3D(
                db_path=self.sfm_info["db_path"],
                new_image_names=new_imgs,
                input_points3D_path=points_path,
                output_points3D_path=new_points_out,
            )
            print(f"Only {num_points} of those were new...")
            duration += time.time() - post_process_start

        return num_points, duration

    def _run_semantic_detection(self, pc_dir):
        """
        Starts semantic_detection on a different process to detect dynamic objects on the newly triangulated point cloud.
        """
        root_dir = Path(__file__).resolve().parent
        parent_dir = root_dir.parent
        detect_config = os.path.abspath(
            os.path.join(parent_dir, self.config["detection"]["detection_config"])
        )
        pc_dir = os.path.abspath(pc_dir)
        cmd = [sys.executable,
               "-m",
               self.config["detection"]["detection_module"],
               pc_dir,
               detect_config,
               "--db_path", self.sfm_info["db_path"],
               "--mask_dir", self.sfm_info["masks_dir"],
               ]
        print(f"Calling detection with {cmd}")

        if self.config['debug']:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=parent_dir
            )

            for l in process.stdout:
                print(l, end="")

            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(...)
        else:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=parent_dir
            )

    def _cleanup(self):
        try:
            self.new_images_txt.close()
            self.matches_txt.close()
        except Exception as e:
            print(e)

    def _get_next_batch(self) -> List[Tuple[Client, List[str]]]:
        """
        Returns next batch of images for each active stream.

        Output:
            [(stream, [img1, img2, ...]), ...]

        Behavior:
        - Pulls up to batch_size images per stream
        - If batch_size == -1 → return all remaining images
        - Automatically unregisters exhausted streams
        """
        new_batches: List[Tuple[Client, List[str]]] = []

        # copy since might remove empty streams
        streams = list(self.image_streams.values())

        for stream in streams:
            # empty stream case
            if stream.cursor >= len(stream.images):
                self._unregister_stream(stream)
                continue
            # otherwise determine batch slice
            if self.batch_size == -1:
                batch = stream.images[stream.cursor:]
                stream.cursor = len(stream.images)
            else:
                end = min(stream.cursor + self.batch_size, len(stream.images))
                batch = stream.images[stream.cursor:end]
                stream.cursor = end
            if batch:
                new_batches.append((stream, batch))
            if stream.cursor >= len(stream.images):
                self._unregister_stream(stream)

        return new_batches

    def register_stream(self, stream: Client):
        self.image_streams[stream.images_path] = stream

    def _unregister_stream(self, stream: Client):
        if stream.images_path in self.image_streams:
            del self.image_streams[stream.images_path]

    def _get_point_cloud_dir(self, batch_id):
        d = os.path.join(self.sfm_info["point_cloud_dir"], f"iteration{batch_id}")
        os.makedirs(d, exist_ok=True)
        return d

    print_skip = lambda x: (f"==================== Batch {x} ====================\n"
                            f"Skipped - not enough streams\n"
                            f"=========================================================\n")
    get_batch_log = lambda batch_id, batch: {
        "batch_id": batch_id,
        "num_streams": len(batch),
        "frames": 0,
        "timings": {
            "extraction": 0.0,
            "matching_colmap": 0.0,
            "matching_proxy": 0.0,
            "verification": 0.0,
            "registration": 0.0,
            "triangulation": 0.0,
            "total": 0.0
        },
        "counts": {
            "colmap_frames": 0,
            "proxy_frames": 0,
            "points_added": 0
        }
    }

def print_final_summary(
        log_data: dict,
):
    totals = log_data["totals"]
    extraction = totals["extraction"]
    matching_colmap = totals["matching_colmap"]
    matching_proxy = totals["matching_proxy"]
    verification = totals["verification"]
    registration = totals["registration"]
    triangulation = totals["triangulation"]
    total_time = totals["total_time"]
    points_added = totals["points_added"]
    print(f"\n=========================================================")
    print(f"Incremental SFM Summary:")
    print(f"Number of Streams: {log_data['num_streams']}")
    print(f"\nLatency Breakdown (Total vs Per-Frame):")
    print(f" • Feature Extraction:   {extraction:>8.2f}s")
    print(f" • Matching (colmap):    {matching_colmap:>8.2f}s")
    print(f" • Matching (custom):    {matching_proxy:>8.2f}s")
    print(f" • Verification:        {verification:>8.2f}s")
    print(f" • Registration:        {registration:>8.2f}s")
    print(f" • Triangulation:       {triangulation:>8.2f}s")
    print(f"\nEnd-to-end Latency:")
    print(f"{total_time:>8.2f}s")
    print(f"\nReconstruction Stats:")
    print(f" • Points Added: {points_added}")
    print(f"=========================================================\n")

def print_batch_summary(
        batch_log: dict,
):
    batch_id = batch_log["batch_id"]
    num_streams = batch_log["num_streams"]
    count = max(1, batch_log["frames"])

    timings = batch_log["timings"]
    counts = batch_log["counts"]

    extraction_time = timings["extraction"]
    matching_colmap_time = timings["matching_colmap"]
    proxy_time = timings["matching_proxy"]
    verification_time = timings["verification"]
    registration_time = timings["registration"]
    triangulate_time = timings["triangulation"]
    duration = timings["total"]

    colmap_count = counts["colmap_frames"]
    proxy_count = counts["proxy_frames"]
    points_added = counts["points_added"]

    proxy_per_frame = proxy_time / proxy_count if proxy_count > 0 else 0
    colmap_per_frame = matching_colmap_time / colmap_count if colmap_count > 0 else 0

    s = f"""
        ==================== Batch {batch_id} ====================
        Frames processed: {count}
        Frames from {num_streams} streams
        Latency Breakdown (this batch):
         • Feature Extraction: {extraction_time:>7.2f}s | {extraction_time / count:>7.3f}s/fr
         • Matching (colmap) {colmap_count} frames: {matching_colmap_time:>7.2f}s | {colmap_per_frame:>7.3f}s/fr
         • Matching (custom) {proxy_count} frames: {proxy_time:>7.2f}s | {proxy_per_frame:>7.3f}s/fr
         • Verification: {verification_time:>7.2f}s
         • Registration: {registration_time:>7.2f}s | {registration_time / count:>7.3f}s/fr
         • Triangulated {points_added} new points: {triangulate_time:>7.2f}s

        Total Batch Time: {duration:>7.2f}s | {duration / count:>7.3f}s/fr
        =========================================================
        """
    print(s)