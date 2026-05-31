# Background & Objectives

I recently came across a session by S&P Global on their use of GNNs & Nvidia Cugraph in a use case where the firm explores the ability to predict issuance of bonds (https://www.nvidia.com/en-us/on-demand/session/gtc25-s74726/) and due to the nature of the firm being a major data provider to most wall street shops on various packages and sectors, I figured that the team would certainly like to verify that the GNN model actually learns the underlying causal financial mechanisms instead of merely exploiting superficial structural correlations.

The graph being a large heterogeneous graph connecting companies, bonds, markets and clients presents a complex prediction challenge as the GNN model has to capture interaction between micro-level corporate needs (liquidity, debt financing) and macro-level market dynamics (investor demand, yeild curves..etc). 

This requires a sophisticated architecture that learns from node features as well as local & global neighborhood structures. Using any models that simply pool & aggregate features like GCNs or GraphSage would likely not fit the bill here as they'll oversmooth embeddings and perform poorly on smaller issuers or companies within the same large sectors (e.g. tech) where the companies share nearly identical maro exposures, patterns and balance sheets profiles.

Attenion is needed here to actually learn a weight for each connection an be able to differenciate between companies, sectors and different time periods.

Graph Attention Networks (GAT) in particular with their multi-attention heads can zero in on tracking independant financial dynamics simultaneously. Head 1 could track macro market liquidity, Head 2 can track balance sheet health...etc.

But how would we explain the results? For a prediction on a company issuing a bond, how can say with (some sort of) confidence that the bond is issued because of node features or because of structural similarities?

For this experiment I thought to take a deeper look at XAI (explainable AI methods) and try to understand if the attention coefficients learned match integrated gradients (IG), which are typically associated to influence.

The hypothesis here is if we had edges with a high IG and high attention, it would mean that the model has successfully learned local features with causal reality. If not, then the model has likely only learned structural redundancies.

To conduct this experiment, we'll be using CAPTUM (https://captum.ai/tutorials/Titanic_Basic_Interpret) a pytorch tool used to interpret models using integraded gradients. Read more on CAPTUM & IGs here (https://arxiv.org/pdf/2009.07896)


## Important Note

Github had recent rendering issues on jupyter notebooks, if you cannot see the [notebook](/GAT%20Attention%20vs%20Influence.ipynb) properly, there's a rendered [markdown version](/GAT%20Attention%20vs%20Influence.md) in the same location.
