## CMN metric 
Code for the paper "Evaluating open-domain dialogues in latent space with next sentence prediction and mutual information" ACL2023

Model can be found at this google drive folder https://drive.google.com/drive/u/1/folders/1qkqcJCprrBaLBdxgoP3dVuD_QInApLk1

Download checkpoints and move to the corresponding datasets folders (dd, personachat)
### Training commands
```
python Training.py --datasets [dd, personachat]
```
### Evaluation commands
```
python Evaluation.py --datasets [dd, personachat]
```

### Citation
If you found this repository or paper is helpful to you, please cite our paper.
```
@article{zhao2023evaluating,
  title={Evaluating Open-Domain Dialogues in Latent Space with Next Sentence Prediction and Mutual Information},
  author={Zhao, Kun and Yang, Bohao and Lin, Chenghua and Rong, Wenge and Villavicencio, Aline and Cui, Xiaohui},
  journal={arXiv preprint arXiv:2305.16967
        
        },
  year={2023}
}
```
