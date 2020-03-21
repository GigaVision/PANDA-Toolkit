## Notice

The toolkit is currently in beta version, has not been extensively tested, and may have bugs.
If you find any problems during use, please put an issue or contact the [author](mailto:wangxuey19@mails.tsinghua.edu.cn).

Many thanks to  [DOTA dataset devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit).


## Functions

The code is useful for <a href="http://www.panda-dataset.com/">PANDA<a> or
<a href="https://www.biendata.com/competition/gigavision/">GigaVision Challenge<a>. The code provide the following function

<ul>
    <li>
        Load and image, and show the bounding box on it.
    </li>
    <li>
        Evaluate the result.
    </li>
    <li>
        Split and merge the picture and label.
    </li>
</ul>

### Installation
1. Environment: **Python 3**
2. Get PANDA from [download page](http://www.panda-dataset.com/Download.html).
3. Install dependencies
```
    pip install -r requirements.txt
```
### Usage
1. Please see tool kit function demonstration in "demo.py"
2. Reading and visualizing data, you can use "PANDA.py"
3. Evaluating the result, you can refer to the "DetEval.py" and "MOTEval.py" 
4. Split the large image, you can refer to the "ImgSplit.py"
5. Merging the results detected on the patches, you can refer to the "ResultMerge.py"

