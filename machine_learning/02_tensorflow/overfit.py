import tensorflow as tf
import pathlib
import shutil
import tempfile

if __name__ == '__main__':
    print(tf.__version__)

    logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
    shutil.rmtree(logdir, ignore_errors=True)

    gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')

    pass
