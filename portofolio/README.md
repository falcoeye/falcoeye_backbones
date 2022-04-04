# Models


## Fish Models:

A set of models that were trained on the Bhader dataset (see datasets)

### Models

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Task</th>
<th valign="bottom">Good For</th>
<th valign="bottom">Framework</th>
<th valign="bottom">Base Architecture</th>
<th valign="bottom">Image Size</th>
<th valign="bottom">Extra</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<tr><td align="left">KAUST Fourty Three Fish SSD</td>
<td align="center">detection</td>
<td align="center">monitoring fish</td>
<td align="center">tensorflow</td>
<td align="center">ssd</td>
<td align="center">1024X1024</td>
<td align="center">SSD ResNet50 V2 1024x1024</td>
<td align="center"><a href="https://drive.google.com/drive/folders/1J368Vx3mZNZajD2ZnwP7J4SLgPhjTEC6?usp=sharing">Frozen Model</a>
</tr>

<tr><td align="left">KAUST Fourty Three Fish FRCNN</td>
<td align="center">detection</td>
<td align="center">monitoring fish</td>
<td align="center">tensorflow</td>
<td align="center">frcnn</td>
<td align="center">1024X1024</td>
<td align="center">FRCNN ResNet50 V2 1024x1024</td>
<td align="center"><a href="https://drive.google.com/drive/folders/15oixkwv88kcylVal97KS6hrPOgIwqltq?usp=sharing">Frozen Model</a>
</tr>




<tr><td align="left">KAUST Counter</td>
<td align="center">detection</td>
<td align="center">counting fish</td>
<td align="center">tensorflow</td>
<td align="center">frcnn</td>
<td align="center">1024X1024</td>
<td align="center">Faster R-CNN ResNet50 V1 1024x1024</td>
<td align="center"><a href="https://drive.google.com/file/d/19MaQEJA2nBZ-0G2y8jCUtyzHmZ86se3p/view?usp=sharing">ckpt</a>,<a href="https://drive.google.com/file/d/1jnshjLyyeivo7eIPsC6xiHD8wXQtMAhF/view?usp=sharing">index</a> | <a href="https://drive.google.com/file/d/19iOGoZPv80_ZXI2I-4leFiiVRc7GkBNH/view?usp=sharing">Labels</a> | <a href="https://drive.google.com/file/d/13VSl5Ts3ejymmGbEv_ShbpXDF65CM8WD/view?usp=sharing">pipeline</a></td>
</tr>

<tr><td align="left">Bhader Counter</td>
<td align="center">detection</td>
<td align="center">counting fish</td>
<td align="center">tensorflow</td>
<td align="center">frcnn</td>
<td align="center">1024X1024</td>
<td align="center">Faster R-CNN ResNet50 V1 1024x1024</td>
<td align="center"><a href="https://drive.google.com/file/d/1OgimZoe2TLIMyxJPtFwho_LlvV77PCrI/view?usp=sharing">ckpt</a>,<a href="https://drive.google.com/file/d/1XkZ6qX8fA9ETgblbxkOcGRCa_HnYc7YC/view?usp=sharing">index</a> | <a href="https://drive.google.com/file/d/1FYSY8pSRUyjymD8F-0TOldf2h9SoDxkc/view?usp=sharing">Labels</a> | <a href="https://drive.google.com/file/d/1_SUehH0zq_klamzJAliOOSNOzvhaw4Ib/view?usp=sharing">pipeline</a></td>
</tr>


<tr><td align="left">Bhader Recognizer</td>
<td align="center">detection</td>
<td align="center">recognizing fish</td>
<td align="center">tensorflow</td>
<td align="center">frcnn</td>
<td align="center">1024X1024</td>
<td align="center">Faster R-CNN ResNet50 V1 1024x1024</td>
<td align="center"><a href="https://drive.google.com/file/d/13wYpNnEm59v_lxXjl5HzBbSFh8Vtsok4/view?usp=sharing">ckpt</a>,<a href="https://drive.google.com/file/d/1o92N4QepP6qo9F6OdAUh7QSRiNkgUc9m/view?usp=sharing">index</a> | <a href="https://drive.google.com/file/d/1VGNkbWCs6R2pPtCHZvcY-exzu8pZeDGT/view?usp=sharing">Labels</a> | <a href="https://drive.google.com/file/d/1FTHkITpe36SHCPYyPt7AxzMpmheR-bim/view?usp=sharing">pipeline</a></td>
</tr>


<tr><td align="left">Bhader Super Counter</td>
<td align="center">detection</td>
<td align="center">counting fish</td>
<td align="center">detectron2</td>
<td align="center">frcnn</td>
<td align="center">min(640, 672, 704, 736, 768, 800)</td>
<td align="center">COCO-Detection/faster_rcnn_R_50_FPN_3x.</td>
<td align="center"><a href="https://drive.google.com/file/d/10JzTmOCP_aOHtDfz9_dVhLthHhMBBgKV/view?usp=sharing">Weights</a> | <a href="https://drive.google.com/file/d/1MrpBh84ZBsJo3vJLraWNys22vM-aQSXi/view?usp=sharing">Labels</a></td>
</tr>


<tr><td align="left">Bhader Super Recognizer</td>
<td align="center">detection</td>
<td align="center">recognizing fish</td>
<td align="center">detectron2</td>
<td align="center">frcnn</td>
<td align="center">min(640, 672, 704, 736, 768, 800)</td>
<td align="center">COCO-Detection/faster_rcnn_R_50_FPN_3x.</td>
<td align="center"><a href="https://drive.google.com/file/d/1sABh7YUQIV06NAHkMZKTipsJ1NWTdtku/view?usp=sharing">Weights</a> | <a href="https://drive.google.com/file/d/1pl5U0Lkfqpt5mVOfy8Ouh2X76FcNeNbI/view?usp=sharing">Labels</a></td>
</tr>



</tbody></table>


## Vehicles Models:

A set of models for vehicles detection

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Task</th>
<th valign="bottom">Good For</th>
<th valign="bottom">Framework</th>
<th valign="bottom">Base Architecture</th>
<th valign="bottom">Image Size</th>
<th valign="bottom">Extra</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<tr><td align="left">careye</td>
<td align="center">detection</td>
<td align="center">monitoring 4 types of vehicles [two-wheels|four-wheels|bus|truck]</td>
<td align="center">torch</td>
<td align="center">yolov5</td>
<td align="center">unknown</td>
<td align="center"></td>
<td align="center"><a href="https://drive.google.com/drive/folders/1hIPhIaQMRrRBok6uVANLFRtwIfLTxlVI?usp=sharing">Model</a>
</tr>


</tbody></table>
