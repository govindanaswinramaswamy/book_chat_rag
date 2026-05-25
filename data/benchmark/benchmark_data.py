from langchain_core.messages import HumanMessage, AIMessage


benchmark_data = [

    {
        "chat_history": [],
        "question": "What renewed the interest of Machine Learning?",
        "answer": "The incorporation of Google's multilingual neural machine translation system into Google Translate in November 2016 renewed interest in machine learning.",
        "group": "lookup"
    },

    {
        "chat_history": [],
        "question": "What type of features are typically produced through batch processing and stream processing in ML systems?",
        "answer": "Batch processing typically produces static features that change slowly, while stream processing produces dynamic features that change rapidly in real time.",
        "group": "lookup"
    },

    {
        "chat_history": [],
        "question": "What is stratified sampling and why is it used?",
        "answer": "Stratified sampling is a method where the population is divided into groups (strata) and samples are taken from each group separately. It is used to ensure that all important groups are represented in the sample.",
        "group": "lookup"
    },

    {
        "chat_history": [],
        "question": "How do machine learning systems differ from traditional software systems, and what challenges do they introduce?",
        "answer": "Machine learning systems differ from traditional software systems in both how they are built and the challenges they introduce. Traditional software typically relies on explicitly programmed rules and modular separation between code and data, whereas ML systems learn patterns directly from data. This makes ML especially useful for complex tasks such as object detection or language understanding, where manually defining rules would be impractical. Unlike research-focused ML, production ML systems must operate reliably under real-world constraints such as scalability, latency, monitoring, and changing data distributions. As a result, ML systems require specialized tooling for data management, model training, deployment, evaluation, and continuous improvement. Because behavior is driven by both code and learned parameters from data, ML systems are often referred to as Software 2.0.",
        "group": "conceptual"
    },

    {
        "chat_history": [],
        "question": "What is the difference between multiclass and multilabel classification, and what challenges does multilabel classification introduce?",
        "answer": "In multiclass classification, each example belongs to exactly one class, whereas in multilabel classification an example can belong to multiple classes simultaneously. Multilabel problems can be handled either by using multi-hot label representations or by training separate binary classifiers for each class. However, multilabel classification is more challenging because annotation can be inconsistent when annotators disagree on applicable labels, and prediction is harder since the model must determine how many labels to assign rather than selecting just one, making probability interpretation and thresholding more complex.",
        "group": "conceptual"
    },

    {
        "chat_history": [],
        "question": "What is sampling in machine learning, why is it important, and what are its main types?",
        "answer": "Sampling is important in machine learning for managing data size, reducing bias, and improving efficiency. The two main families are nonprobability sampling (e.g., convenience sampling) and random sampling (e.g., reservoir sampling and importance sampling). Effective sampling helps create more representative datasets while optimizing resource usage.",
        "group": "conceptual"
    },

    {
        "chat_history": [],
        "question": "What are the four key requirements of a production ML system, and what does each requirement mean?",
        "answer": "Most ML systems should satisfy four key requirements: reliability, scalability, maintainability, and adaptability. Reliability means the system continues functioning correctly even under failures or adverse conditions. Scalability allows the system to handle growth in data, traffic, and model complexity. Maintainability supports collaboration, reproducibility, and easier debugging through structured workflows and tooling. Adaptability enables the system to evolve as data distributions and business requirements change over time.",
        "group": "long-context"
    },

    {
        "chat_history": [],
        "question": "How do batch processing and stream processing differ, what types of ML features does each support, and why is stream processing important in modern ML systems?",
        "answer": "Batch processing operates on historical data stored in databases, data lakes, or warehouses and typically runs periodically using systems like MapReduce or Spark. It is commonly used to compute slowly changing or static features, such as driver ratings. Stream processing operates on continuously arriving data from systems like Kafka or Kinesis and processes data in real time or near real time using engines such as Apache Flink, KSQL, or Spark Streaming. It is used to compute rapidly changing or dynamic features, such as live demand, available drivers, or recent prices. Stream processing is important because it enables low-latency computation, supports stateful incremental updates without repeatedly recomputing entire datasets, and allows ML systems to react quickly to changing conditions. Modern ML systems often combine both batch and streaming features, requiring infrastructure capable of processing and joining both data types.",
        "group": "long-context"
    },

    {
        "chat_history": [],
        "question": "What factors determine whether an ML feature generalizes well to unseen data, and what trade-offs can arise when designing more generalizable features?",
        "answer": "A feature generalizes well when it remains useful on unseen data instead of only fitting the training data. Important factors include feature coverage and whether feature values are consistent across train and test data. Features that are too specific may improve accuracy on training data but perform poorly on new data.",
        "group": "long-context"
    },

    {
        "chat_history": [],
        "question": "Who is Aristotle?",
        "answer": "I cannot answer this question as it is out of my knowledge.",
        "group": "external"
    },

    {
        "chat_history": [],
        "question": "What is inflation?",
        "answer": "I cannot answer this question as it is out of my knowledge.",
        "group": "external"
    },

    {
        "chat_history": [],
        "question": "Why did the Industrial Revolution lead to urbanization?",
        "answer": "I cannot answer this question as it is out of my knowledge.",
        "group": "external"
    },

    {
        "chat_history": [],
        "question": "wht renewed intrest in machine learning again",
        "answer": "The incorporation of Google's multilingual neural machine translation system into Google Translate in November 2016 renewed interest in machine learning.",
        "group": "rephrase-typo"
    },

    {
        "chat_history": [],
        "question": "wat kinda features come from batch and stream processing usually?",
        "answer": "Batch processing typically produces static features that change slowly, while stream processing produces dynamic features that change rapidly in real time.",
        "group": "rephrase-typo"
    },

    {
        "chat_history": [],
        "question": "why stratified sampeling use by ppl?",
        "answer": "Stratified sampling is a method where the population is divided into groups (strata) and samples are taken from each group separately. It is used to ensure that all important groups are represented in the sample.",
        "group": "rephrase-typo"
    },

    {
        "chat_history": [
            HumanMessage(content="Explain batch processing in ML systems."),
            AIMessage(content="Batch processing operates on historical stored data and usually runs periodically."),

            HumanMessage(content="What about stream processing?"),
            AIMessage(content="Stream processing handles continuously arriving real-time data."),

            HumanMessage(content="Why is stream processing important?"),
            AIMessage(content="It helps ML systems react quickly to changing conditions with low latency.")
        ],
        "question": "What type of features do both usually produce?",
        "answer": "Batch processing typically produces static features that change slowly, while stream processing produces dynamic features that change rapidly in real time.",
        "group": "rephrase-history"
    },

    {
        "chat_history": [
            HumanMessage(content="Why are modern ML systems difficult to maintain?"),
            AIMessage(
                content="ML systems are harder to maintain because they depend on both code and data distributions."),

            HumanMessage(content="What makes debugging difficult?"),
            AIMessage(content="Model behavior can be difficult to interpret and reproduce."),

            HumanMessage(content="Tell me about production ML systems."),
            AIMessage(
                content="Production ML systems need to be designed carefully to work reliably in real-world environments.")
        ],
        "question": "What are the four key requirements for them?",
        "answer": "Most ML systems should satisfy four key requirements: reliability, scalability, maintainability, and adaptability. Reliability means the system continues functioning correctly even under failures or adverse conditions. Scalability allows the system to handle growth in data, traffic, and model complexity. Maintainability supports collaboration, reproducibility, and easier debugging through structured workflows and tooling. Adaptability enables the system to evolve as data distributions and business requirements change over time.",
        "group": "rephrase-history"
    },

    {
        "chat_history": [
            HumanMessage(content="What is regression in machine learning?"),
            AIMessage(content="Regression predicts numerical values for given features."),

            HumanMessage(content="Explain multiclass classification."),
            AIMessage(content="Multiclass classification assigns exactly one class to each example."),

            HumanMessage(content="What about multilabel classification?"),
            AIMessage(content="Multilabel classification allows multiple labels for the same example.")
        ],
        "question": "What challenges does the second classification introduce?",
        "answer": "Multilabel classification introduces challenges because annotation can be inconsistent when annotators disagree on applicable labels, and prediction is harder since the model must determine how many labels to assign rather than selecting just one, making probability interpretation and thresholding more complex.",
        "group": "rephrase-history"
    }
]
