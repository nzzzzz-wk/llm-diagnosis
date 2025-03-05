# Study note
## prompt
- 对json格式限制时，prompt中使用的格式 务必使用双引号，避免在json提取时，单引号导致json读取失败
- 存在往多疾病的病例，使用```{"diseases":"text", "reason":"text"}``` 保存结果时，可能会出现多条，但是参赛提交结果示例中，只有一个位置存放结果，容易造成提取上的问题。prompt强调返回一条 有效，也可以不限制json格式，使用两个llm节点的结果文本，节点外合并为json，但会增加额外token
