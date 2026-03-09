# [CVPR 2026] Physical Simulator In-the-Loop Video Generation

<div>
    <span>
        <a href="https://lingeng.foo/" target="_blank" rel="noopener noreferrer">Lin Geng
            Foo</a><sup>1,5</sup>,
    </span>
    <span>
        <a href="https://markhh.com/" target="_blank" rel="noopener noreferrer">Mark He
            Huang</a><sup>2,3</sup>,
    </span>
    <span>
        <a href="https://alexlattas.com/" target="_blank" rel="noopener noreferrer">Alexandros
            Lattas</a><sup>4</sup>,
    </span>
    <span>
        <a href="https://moschoglou.com/" target="_blank" rel="noopener noreferrer">Stylianos
            Moschoglou</a><sup>4</sup>,
    </span>
    <span>
        <a href="https://thabobeeler.com/" target="_blank" rel="noopener noreferrer">Thabo
            Beeler</a><sup>4</sup>,
    </span>
    <span>
        <a href="https://people.mpi-inf.mpg.de/~theobalt/" target="_blank" rel="noopener noreferrer">Christian Theobalt</a><sup>1,5</sup>
    </span>
</div>

<div>
    <span><sup>1</sup>Max Planck Institute for Informatics, Saarland Informatics Campus</span>,
    <span><sup>2</sup>SUTD</span>,
    <span><sup>3</sup>A*STAR</span>,
    <span><sup>4</sup>Google</span>,
    <span><sup>5</sup>Saarbrücken Research Center for
         Visual Computing, Interaction and Artificial Intelligence</span>
</div>

<br>

[[Project Page]](https://vcai.mpi-inf.mpg.de/projects/PSIVG/) | [[Paper]](https://arxiv.org/abs/2603.06408)

![](assets/fig1.jpg)

## Usage

Please refer to the [instructions.md](instructions.md) for environment setup and pipeline execution.

_**Disclaimer**: This repository offers a reference implementation of the pipeline described in the paper. The code is intended for research purposes only, is not optimized for robustness or production environments, and may not be actively maintained._

## License

This project is released under the [Apache 2.0 License](LICENSE).

## BibTeX

```bibtex
@article{foo2026physical, 
  title={Physical Simulator In-the-Loop Video Generation}, 
  author={Foo, Lin Geng and Huang, Mark He and Lattas, Alexandros and Moschoglou, Stylianos and Beeler, Thabo and Theobalt, Christian}, 
  journal={arXiv preprint arXiv:2603.06408}, 
  year={2026} 
}
```

<!-- ```
@InProceedings{Foo_2026_CVPR,
  title={Physical Simulator In-the-Loop Video Generation},
  author={Foo, Lin Geng and Huang, Mark He and Lattas, Alexandros and Moschoglou, Stylianos and Beeler, Thabo and Theobalt, Christian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}  
@misc{foo2026physicalsimulatorintheloopvideo,
      title={Physical Simulator In-the-Loop Video Generation},
      author={Lin Geng Foo and Mark He Huang and Alexandros Lattas and Stylianos Moschoglou and Thabo Beeler and Christian Theobalt},
      year={2026},
      eprint={2603.06408},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.06408},
}
``` -->

## Acknowledgments

This project is built upon invaluable open-source research projects: [ViPE](https://github.com/nv-tlabs/vipe), [PhysGen3D](https://github.com/by-luckk/PhysGen3D), [Taichi](https://github.com/taichi-dev/taichi), [Taichi Elements](https://github.com/taichi-dev/taichi_elements), [Mitsuba](https://github.com/mitsuba-renderer/mitsuba3), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [SAM](https://github.com/facebookresearch/sam2), [LaMa](https://github.com/advimman/lama), [InstantMesh](https://github.com/TencentARC/InstantMesh), [lang-segment-anything](https://github.com/luca-medeiros/lang-segment-anything), [Go-with-the-Flow](https://github.com/Eyeline-Labs/Go-with-the-Flow), [cogvideox-factory](https://github.com/RyannDaGreat/cogvideox-factory), [CogVideoX](https://github.com/zai-org/CogVideo), [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo).

We also thank these research projects for sharing their code: [pisa-experiments](https://github.com/vision-x-nyu/pisa-experiments), [MotionClone](https://github.com/LPengYang/MotionClone), [SG-I2V](https://github.com/Kmcode1/SG-I2V), [ImageConductor](https://github.com/liyaowei-stu/ImageConductor), [DragAnything](https://github.com/showlab/DragAnything), [VBench](https://github.com/Vchitect/VBench).
