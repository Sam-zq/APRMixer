# 上传 GitHub

上传本目录的内容即可，不要上传外层论文工作目录。仓库名建议 `APRMixer`。

## 网页上传

1. 在 GitHub 创建仓库。
2. 解压交付的 ZIP，进入 `APRMixer_GitHub` 文件夹。
3. 将其中的源码目录、README、依赖清单和 `.gitignore` 上传到仓库根目录。
4. 选择作者代码许可证，并补充真实作者和论文引用信息。

## Git 上传

在本目录执行，替换远程地址为你自己的仓库地址：

```bash
git init
git add .
git diff --cached --stat
git commit -m "Add APRMixer model, training and evaluation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/APRMixer.git
git push -u origin main
```

`.gitignore` 已排除数据目录中的原始数据、运行结果、模型权重、预测数组和环境文件。保留第三方许可证文件。原创模型的顶层许可证尚未替你选择。

默认训练入口采用保留原实验初始化顺序的版本。精简模型用于阅读和推理，不能凭相同随机种子声称重现原训练结果。英文 README 中已说明这一点。
