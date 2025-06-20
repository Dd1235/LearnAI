# CaLM
[] implementation is still to do. Will complete by Sunday.

# Attempt at Understanding the Paper

CALM: Composition of Augment Language Models
- compose 2 llms to enable newer capabilities


m_A : Augmenting Model  eg PaLM2-XXS model further trained on domain speicifc data
m_B : Anchor LLM m_B eg PaLM-XS or PaLM2-S.

m_B are larger than m_A. m_A is trained on domain specific data. eg, say m_A knows x1 = 10, x2 = 20, m_B knows addition, then m_A can be used to augment m_B. 

eg. m_A only knows hte key-value mappins and  m_B knows the operations, then m_A can be used to augment m_B.
- This is acheived through cross attention architecture. Take say every 4th layer from m_A, and every foruth layer from m_B, use the key-value from m_A, and query from m_B, and get cross attention outputs that are added via residual connections to m_B.

- m_A is trained on low resource languages, so low resource language to english translation, m_B is a generic model so augment using m_A.

- m_B + further training, gives better reuslts than calm which is better than using mB alone or mA with further training. But hte compute pwoer is calm is much less than mB + further training. (for NTL)

- so mB the general model will ask the Query, mA will provide some context in the form of K and V, and the cross attention will be used, so mA attends to the query from mB. 

- Basically, sometimes further training will cause catastrophic forgetting, so it start to do bad, but in calm the ma and mb weights are frozen so you won't forget the previous knowledge.

But on RAM you need to maintain both the models, train both the models while updating to keep against model drift, could end up being more computationally expensive? Deployment is also more expensive?

# Thoughts

Okay so the matching networks paper was on using a non parametric model. I found the use of cross-attention in CALM reminiscent of the way Matching Networks perform cosine attention between support and query embeddings. Matching networks and Ram Sirs FSL ASR are both related to representation based composition. It's like the same low resource setting, but this is at a more architectural level. Few shot phenome recognition dealt with comparing audio embedding to support set audio embeddings, and pick the closest phenome class.

Cross attention is new here, but I wonder why not bidirectional cross attention? why only the augmenting model attends to the anchor model? Maybe some context from teh anchor model would improve the augmenting model? How does it compare to RAG? 

Does it make sense to maintain a bunch of different augmenting models that specialize and combine them? The way human expertise is usually employed in achieving a task.

All these seem very biologically inspired.

# Look into

[Project Vaani, lots of speech data for Indian langauges](https://vaani.iisc.ac.in/#Data)

I find it funny that every state has so many hours of different languages, like AP has more Hindi than Telugu, there is some Bengali etc, but Tamil Nadu has only Tamil and English.