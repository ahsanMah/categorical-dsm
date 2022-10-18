from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import tensorflow as tf
import wandb
import ml_collections
from pprint import pprint
import runner

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)

flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum(
    "mode", None, ["train", "eval", "score", "sweep"], "Running mode: train or eval"
)
flags.DEFINE_string(
    "eval_folder", "eval", "The folder name for storing evaluation results"
)
flags.DEFINE_string(
    "sweep_id", None, "Optional ID for a sweep controller if running a sweep."
)
flags.DEFINE_string("project", None, "Wandb project name.")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
    config = FLAGS.config
    workdir = FLAGS.workdir
    runner.train(config, workdir)


if __name__ == "__main__":
    app.run(main)
