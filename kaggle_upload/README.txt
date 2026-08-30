KAGGLE UPLOAD -- everything you need, nothing you don't
=======================================================

Two files in this folder. They go to two DIFFERENT places on Kaggle.


1_DATASET_neuron-death-code.zip
    -> https://www.kaggle.com/datasets   ("New Dataset", or "New Version" of
       your existing neuron-death-code dataset)
    Upload the .zip as-is. Kaggle unzips it for you.
    It already contains mnist.npz, so this is the ONLY dataset you need to
    attach. You do not need a separate MNIST dataset.
    UPLOAD THIS FIRST, and re-upload it whenever the code or configs change.


2_*.ipynb ... 11_*.ipynb  -- one notebook per experiment
    -> https://www.kaggle.com/code   ("New Notebook" -> File -> Import Notebook)

    These are INDEPENDENT. No experiment reads another's output and every
    learning rate is already fixed by the gate, so they can all run at the same
    time in separate sessions. Nothing inside them needs editing.

    Priority order if you run them one at a time:
      2_neuron_methods         <- current professor-requested next step
      3_tau_a_none_redo
      4_tau_b_random_matched
      5_setting3_activations
      6_c5_optimizers
      7_tau_c_inverse_matched
      8_c3_anomaly
      9_setting3_tanh_gate
      10_setting2_gate
      11_setting2_cifar_cnn    <- needs CIFAR-10, see below


THEN, in each notebook:
    - Add Input -> attach the dataset from step 1
    - Accelerator -> GPU T4 x2
    - Run All


PRE-FLIGHT CHECK (built in)
    The "Pick the experiment" cell now ASSERTS the config count and stops if it
    is wrong. If it fails, the attached Dataset is a stale version -- re-upload
    step 1 rather than letting the sweep run.


11_setting2_cifar_cnn NEEDS A SECOND DATASET
    It runs on CIFAR-10. Do NOT download or upload it -- search Kaggle for
    "CIFAR-10 python" and attach any public one as a second Input. src/data.py
    reads all three layouts these come in:
        cifar10.npz  |  cifar-10-batches-py/  |  cifar-10-python.tar.gz
    The notebook finds whichever is mounted and asserts if none is.

    Its per-run cost is unmeasured -- a conv net is not an MLP. Read the
    smoke-test cell's printed time and multiply by 15 before running the sweep.


AT THE END
    extract.zip   -> DOWNLOAD this one. A few MB. All the analysis needs.
    runs.zip      -> push to a versioned Dataset. Hundreds of MB. Do not
                     download; it is the archival copy of the per-neuron log.
