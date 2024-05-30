# tree2retriever
面向RAG场景的递归摘要树检索器实现

> Recursive Abstractive Processing for Tree-Organized Retrieval

## Example
```python
import logging
import pickle

from tree2retriever.cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from tree2retriever.embedding import SBertEmbeddingModel
from tree2retriever.summary import GLMSummarizationModel
from tree2retriever.tree_retriever import TreeRetriever, TreeRetrieverConfig

if __name__ == '__main__':
    tree_builder_type = 'cluster'
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
    supported_tree_builders = {"cluster": (ClusterTreeBuilder, ClusterTreeConfig)}

    tree_builder_class, tree_builder_config_class = supported_tree_builders[
        tree_builder_type
    ]
    embedding_model = SBertEmbeddingModel(model_name=r"I:\pretrained_models\bert\english\all-mpnet-base-v2")
    summary_model = GLMSummarizationModel(model_name_or_path=r"I:\pretrained_models\llm\chatglm3-6b")
    tree_builder_config = tree_builder_config_class(
        tokenizer=None,
        max_tokens=100,
        num_layers=5,
        threshold=0.5,
        top_k=5,
        selection_mode="top_k",
        summarization_length=100,
        summarization_model=summary_model,
        embedding_models={'sbert': embedding_model},
        cluster_embedding_model="sbert",
    )

    tree_retriever_config = TreeRetrieverConfig(
        tokenizer=None,
        threshold=0.5,
        top_k=5,
        selection_mode="top_k",
        context_embedding_model="sbert",
        embedding_model=embedding_model,
        num_layers=None,
        start_layer=None,
    )

    tree_builder = tree_builder_class(tree_builder_config)

    with open(r'H:\Projects\GoMate\data\docs\sample.txt', 'r') as file:
        text = file.read()
    tree = tree_builder.build_from_text(text=text)
    retriever = TreeRetriever(tree_retriever_config, tree)
    question = '"How did Cinderella reach her happy ending?'

    search_docs = retriever.retrieve(question)
    print(search_docs)

    path = "tree.pkl"
    with open(path, "wb") as file:
        pickle.dump(tree, file)
    logging.info(f"Tree successfully saved to {path}")

```

运行日志如下：
```text
2024-05-30 21:20:29,655 - Load pretrained SentenceTransformer: I:\pretrained_models\bert\english\all-mpnet-base-v2
2024-05-30 21:20:56,691 - Use pytorch device_name: cuda
Loading checkpoint shards: 100%|██████████| 7/7 [07:04<00:00, 60.64s/it]
2024-05-30 21:28:05,104 - Successfully initialized TreeBuilder with Config 
        TreeBuilderConfig:
            Tokenizer: <Encoding 'cl100k_base'>
            Max Tokens: 100
            Num Layers: 5
            Threshold: 0.5
            Top K: 5
            Selection Mode: top_k
            Summarization Length: 100
            Summarization Model: <gomate.modules.refiner.summary.GLMSummarizationModel object at 0x0000015C2BE18C50>
            Embedding Models: {'sbert': <gomate.modules.retrieval.embedding.SBertEmbeddingModel object at 0x0000015C1B25D790>}
            Cluster Embedding Model: sbert
        
        Reduction Dimension: 10
        Clustering Algorithm: RAPTOR_Clustering
        Clustering Parameters: {}
        
2024-05-30 21:28:05,104 - Successfully initialized ClusterTreeBuilder with Config 
        TreeBuilderConfig:
            Tokenizer: <Encoding 'cl100k_base'>
            Max Tokens: 100
            Num Layers: 5
            Threshold: 0.5
            Top K: 5
            Selection Mode: top_k
            Summarization Length: 100
            Summarization Model: <gomate.modules.refiner.summary.GLMSummarizationModel object at 0x0000015C2BE18C50>
            Embedding Models: {'sbert': <gomate.modules.retrieval.embedding.SBertEmbeddingModel object at 0x0000015C1B25D790>}
            Cluster Embedding Model: sbert
        
        Reduction Dimension: 10
        Clustering Algorithm: RAPTOR_Clustering
        Clustering Parameters: {}
        
2024-05-30 21:28:05,120 - Creating Leaf Nodes
Batches: 100%|██████████| 1/1 [00:01<00:00,  1.50s/it]
Batches: 100%|██████████| 1/1 [00:00<00:00, 58.82it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 66.66it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 68.96it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 66.66it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 68.97it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 64.49it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 62.51it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 54.06it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 60.60it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 68.96it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 71.42it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 68.97it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 57.14it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 57.14it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 52.63it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 47.63it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 68.96it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 68.94it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 68.97it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 68.97it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 68.97it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 66.67it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 62.49it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 66.67it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 47.62it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 60.61it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 64.51it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 62.50it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 66.67it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 62.50it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 66.70it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 66.67it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 66.66it/s]
Batches: 100%|██████████| 1/1 [00:00<00:00, 14.08it/s]
2024-05-30 21:28:07,312 - Created 35 Leaf Embeddings
2024-05-30 21:28:07,312 - Building All Nodes
2024-05-30 21:28:07,314 - Using Cluster TreeBuilder
2024-05-30 21:28:07,314 - Constructing Layer 0
2024-05-30 21:28:14,750 - Summarization Length: 100
C:\Users\yanqiang\.cache\huggingface\modules\transformers_modules\chatglm3-6b\modeling_chatglm.py:226: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at C:\cb\pytorch_1000000000000\work\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:455.)
  context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
2024-05-30 21:28:27,157 - The story is about a rich man's wife who is sick and near death. She advises her daughter to be good and pious, and promises to look down on her from heaven. The daughter is devoted and remains pious, even when her mother passes away.

Every day, the daughter goes to her mother's grave and weeps. One day, the snow covers the grave, but when the sun melts it away, the man has already taken another wife. This new wife brings two daughters who are beautiful but evil-hearted.

The step-child is treated poorly and is told to go out into the garden and pick the good seeds for the dish. The birds, including two white pigeons and turtle-doves, come to help her. The pigeons and turtle-doves gather the good seeds and eat them before the half-hour is over. The step-child is delighted and believes she will now be allowed to go to the wedding.
2024-05-30 21:28:27,158 - Node Texts Length: 563, Summarized Text Length: 187
Batches: 100%|██████████| 1/1 [00:00<00:00, 55.55it/s]
2024-05-30 21:28:34,895 - The story is about a beautiful maiden who is forced to work hard and suffer abuse by her step-mother and two sisters. She is taken away from her pretty clothes and given an old grey bedgown and wooden shoes. The sisters mock and injuries are mentioned, but the maiden remains resilient. One day a prince approaches her and they dance together. The prince wants to see who the beautiful maiden belongs to and ends up following her. She escapes from him and runs into the garden behind the house. There, she finds a tree with the most magnificent pears and climbs it nimbly. The prince becomes anxious to go with her, but she escape quickly. The next day, the prince gives her a golden shoe that fits perfectly. The step-mother and sisters are horrified and become pale with rage.
2024-05-30 21:28:34,895 - Node Texts Length: 477, Summarized Text Length: 162
Batches: 100%|██████████| 1/1 [00:00<00:00, 64.51it/s]
2024-05-30 21:28:38,211 - The two step-sisters are excited about going to the wedding at the king's palace, but Cinderella wants to go with them. However, her step-mother prevents her from going, saying that Cinderella has no clothes and cannot dance. Cinderella cries and goes to her mother's grave to comfort herself.
2024-05-30 21:28:38,212 - Node Texts Length: 194, Summarized Text Length: 63
Batches: 100%|██████████| 1/1 [00:00<00:00, 64.51it/s]
2024-05-30 21:28:47,037 - Cinderella is a young woman who lives in a house with her step-parents. She is treated poorly and is not allowed to attend the king's ball, where she believes she will be able to find true love. Instead, her step-parents go to the ball and Cinderella is left behind. While at the ball, a magical pumpkin transforms into a carriage, and Cinderella is taken to the palace and meets the king's son, who is in search of a bride. Cinderella is able to go to the ball and attend the wedding thanks to the intervention of a magical bird, who throws down golden and silver items to her when she expresses a wish. At the wedding, Cinderella's step-sisters and step-mother do not recognize her, and believe she is a foreign princess. Eventually, Cinderella is discovered and is able to live happily ever after with the king's son.
2024-05-30 21:28:47,037 - Node Texts Length: 614, Summarized Text Length: 182
Batches: 100%|██████████| 1/1 [00:00<00:00, 64.51it/s]
2024-05-30 21:28:55,399 - A summary of the story of Cinderella:

Cinderella is a young woman who is forced to live in the household of her cruel and heartless step-mother. She is mistreated and has little in life, but is kind and kind-hearted. One day, a royal prince invites Cinderella to a ball, but she is unable to go because she has no clothes or shoes. Cinderella's step-mother refuses to let her attend, saying that she will be laughed at. Cinderella is desperate, and manages to pick two dishes of lentils out of the ashes for her step-mother in one hour, earning her ticket to the ball. At the ball, Cinderella is revealed to be a beautiful princess, and the prince marries her. The step-mother is punished for her cruelty, and Cinderella lives happily ever after.
2024-05-30 21:28:55,400 - Node Texts Length: 471, Summarized Text Length: 171
Batches: 100%|██████████| 1/1 [00:00<00:00, 62.50it/s]
2024-05-30 21:29:07,644 - The king asked his two step-daughters what they wanted him to bring back from his journey, and they both wished for beautiful dresses, pearls, and jewels. The king then bought these things for his step-daughters, but on his way home, a hazel twig knocked off his hat. He broke off the branch and took it with him, then gave his step-daughters the gifts they had wished for. He also gave the branch to Cinderella, who thanked him and planted it on her mother's grave. Cinderella then went to the pigeon-house, where the king's son thought she had leapt into it. No one was inside, so he had to have an axe and pickaxe brought to break down the pigeon-house. Cinderella was found laying in the ashes, and when her parents and sisters returned home, she was again found there. She took a beautiful dress from a bird on a little hazel-tree and put on her grey gown. On the third day, Cinderella went to her mother's grave and said to the tree, "Shiver and quiver, my little tree, silver and gold throw down over me."
2024-05-30 21:29:07,645 - Node Texts Length: 477, Summarized Text Length: 232
Batches: 100%|██████████| 1/1 [00:00<00:00, 50.06it/s]
2024-05-30 21:29:23,328 - The story is about a young woman who is determined to find a wife for herself, but she has specific requirements for the perfect bride. She wants someone whose foot fits a certain golden slipper. Her two sisters are excited about the prospect of finding a bride, and they try to help her find the perfect one.

The eldest sister tries on the slipper, but it is too small for her, so her mother cuts off the toe and forces her foot into the shoe. The eldest sister endures the pain and goes out to the king's son, who takes her on his horse as his bride.

However, as they pass by a hazel-tree, the two pigeons sitting on it cry out that the shoe is too small and the true bride is still waiting. The king's son realizes that the eldest sister is not the true bride and takes her back home.

The second sister tries on the shoe and fits it perfectly, but her heel is too large, so her mother cuts off a bit and forces her foot into the shoe. The second sister endures the pain and goes out to the king's son, who takes her on his horse as his bride.

Again, as they pass by the hazel-tree, the two pigeons cry out that the shoe is too small and the true bride is still waiting. The king's son realizes that the second sister is not the true bride and takes her back home.

Therefore, the two sisters are left without finding a bride, and the king's son is left with a false bride who is not suitable for his queen.
2024-05-30 21:29:23,329 - Node Texts Length: 485, Summarized Text Length: 317
Batches: 100%|██████████| 1/1 [00:00<00:00, 57.15it/s]
2024-05-30 21:29:23,349 - Constructing Layer 1
2024-05-30 21:29:23,349 - Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: 1
2024-05-30 21:29:23,350 - Successfully initialized TreeRetriever with Config 
        TreeRetrieverConfig:
            Tokenizer: <Encoding 'cl100k_base'>
            Threshold: 0.5
            Top K: 5
            Selection Mode: top_k
            Context Embedding Model: sbert
            Embedding Model: <gomate.modules.retrieval.embedding.SBertEmbeddingModel object at 0x0000015C1B25D790>
            Num Layers: None
            Start Layer: None
        
2024-05-30 21:29:23,350 - Using collapsed_tree
Batches: 100%|██████████| 1/1 [00:00<00:00, 29.41it/s]
2024-05-30 21:29:23,393 - Tree successfully saved to tree.pkl
A summary of the story of Cinderella:  Cinderella is a young woman who is forced to live in the household of her cruel and heartless step-mother. She is mistreated and has little in life, but is kind and kind-hearted. One day, a royal prince invites Cinderella to a ball, but she is unable to go because she has no clothes or shoes. Cinderella's step-mother refuses to let her attend, saying that she will be laughed at. Cinderella is desperate, and manages to pick two dishes of lentils out of the ashes for her step-mother in one hour, earning her ticket to the ball. At the ball, Cinderella is revealed to be a beautiful princess, and the prince marries her. The step-mother is punished for her cruelty, and Cinderella lives happily ever after.

Cinderella is a young woman who lives in a house with her step-parents. She is treated poorly and is not allowed to attend the king's ball, where she believes she will be able to find true love. Instead, her step-parents go to the ball and Cinderella is left behind. While at the ball, a magical pumpkin transforms into a carriage, and Cinderella is taken to the palace and meets the king's son, who is in search of a bride. Cinderella is able to go to the ball and attend the wedding thanks to the intervention of a magical bird, who throws down golden and silver items to her when she expresses a wish. At the wedding, Cinderella's step-sisters and step-mother do not recognize her, and believe she is a foreign princess. Eventually, Cinderella is discovered and is able to live happily ever after with the king's son.

Then the bird threw a gold and silver dress down to her, and slippers embroidered with silk and silver   She put on the dress with all speed, and went to the wedding   Her step-sisters and the step-mother however did not know her, and thought she must be a foreign princess, for she looked so beautiful in the golden dress They never once thought of cinderella, and believed that she was sitting at home in the dirt, picking lentils out of the ashes   The

And now the bird threw down to her a dress which was more splendid and magnificent than any she had yet had, and the slippers were golden   And when she went to the festival in the dress, no one knew how to speak for astonishment   The king's son danced with her only, and if any one invited her to dance, he said this is my partner When evening came, cinderella wished to leave, and the king's

 Thrice a day cinderella went and sat beneath it, and wept and prayed, and a little white bird always came on the tree, and if cinderella expressed a wish, the bird threw down to her what she had wished for It happened, however, that the king gave orders for a festival which was to last three days, and to which all the beautiful young girls in the country were invited, in order that his son might choose himself a bride

the step-sisters had gone once more, cinderella went to the hazel-tree and said -      shiver and quiver, my little tree,      silver and gold throw down over me Then the bird threw down a much more beautiful dress than on the preceding day  And when cinderella appeared at the wedding in this dress, every one was astonished at her beauty   The king's son had waited until she came, and instantly took her by the hand

and emptied her peas and lentils into the ashes, so that she was forced to sit and pick them out again   In the evening when she had worked till she was weary she had no bed to go to, but had to sleep by the hearth in the cinders   And as on that account she always looked dusty and dirty, they called her cinderella It happened that the father was once going to the fair, and he

a dim little oil-lamp was burning on the mantle-piece, for cinderella had jumped quickly down from the back of the pigeon-house and had run to the little hazel-tree, and there she had taken off her beautiful clothes and laid them on the grave, and the bird had taken them away again, and then she had seated herself in the kitchen amongst the ashes in her grey gown Next day when the festival began afresh, and her parents and

however, took cinderella on his horse and rode away with her   As they passed by the hazel-tree, the two white doves cried -      turn and peep, turn and peep,      no blood is in the shoe,      the shoe is not too small for her,      the true bride rides with you, and when they had cried that, the two came flying down and placed themselves on cinderella's shoulders, one on the right,

kitchen, cinderella lay there among the ashes, as usual, for she had jumped down on the other side of the tree, had taken the beautiful dress to the bird on the little hazel-tree, and put on her grey gown On the third day, when the parents and sisters had gone away, cinderella went once more to her mother's grave and said to the little tree -      shiver and quiver, my little tree,      silver and gold throw down over me
```
## Reference

- RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval