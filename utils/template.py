"""
Description: 
Author: dante
Created on: 2023/11/15
"""

templates = {
    "D_F": {
        "fact_list_template": "Given a passage: \"{sentences}\"\nList all underlying facts."
    },
    "D_RS_F": {
        "relation_list_template": "Given a passage: \"{sentences}\"\nList all underlying relations.",
        "fact_list_template": "Given a passage: \"{sentences}\"\nList all triplet facts according to the relation list : \"{relations}\"n."
    },
    "D_R_F": {
        "relation_list_template": "Given a passage: \"{sentences}\"\nList all underlying relations.",
        "fact_list_template": "Given a passage: \"{sentences}\"\nList all triplet facts that take \"{relation}\" as the relation."
    },
    "D_R_H_F": {
        "relation_list_template": "Given a passage: \"{sentences}\"\nList all underlying relations.",
        "entity_list_template": "Given a relation : \"{relation}\" and a passage: \"{sentences}\"\nlist the entities that can be serve as suitable subjects for the relation.",
        "fact_list_template": "Provided a passage: \"{sentences}\"\nList all triplet facts that take \"{relation}\" as the relation and \"{subject}\" as the subject."
    },
    "D_R_H_F_desc": {
        "relation_list_template": '''Given a passage: "{sentences}"\nList any underlying relations.''',
        "entity_list_template": '''Given a relation "{relation}", and its description: "{description}" and a passage: "{sentences}", list entities that can be identified as suitable subjects for the relation.''',
        "fact_list_template": '''Given relation "{relation}" and relation description: "{description}".\n Provided a passage: "{sentences}"\n List all triplet facts that take "{relation}" as the relation and "{subject}" as the subject.'''
    },
    "D_R_H_F_desc_analysis": {
        "relation_template": "Given a passage: \"{sentences}\", conduct a comprehensive analysis to uncover any underlying relations.",
        "relation_list_template": "Given a passage: \"{sentences}\", along with an analysis of the implicit relations present in the passage: {relation_analysis}.\n"
                                  "List the relations that can be derived from the passage.",
        "entity_template": "Given a description of a relation: \"{description}\" and a passage: \"{sentences}\", analysis which entities can act as the subject of the "
                           "mentioned relation.",
        "entity_list_template": "Given a relation description: \"{description}\" and a passage: \"{sentences}\", along with the analysis of subjects: {subjects_analysis}, "
                                "Based on this analysis, list entities that can be identified as suitable subjects for the relation. ",
        "fact_template": "Given relation \"{relation}\" and relation description: \"{description}\", the passage: \"{sentences}\", and a specific subject: \"{subject}\", "
                         "conduct an analysis to identify all facts within the passage.",
        "fact_list_template": "After analyzing the passage: \"{sentences}\", and considering the subject of the relation: \"{subject}\", we have conducted an analysis to derive "
                              "all facts as: \"{facts_analysis}\".\nBased on this analysis, list the facts."
    },
       "Head_fact": {
        "entity_list_template": '''You are a professional relationship extraction expert. Given a list of relations: "{relation_list}" and a document: "{sentences}", extract the entities that can serve as the subjects of the relations. Output the result as a JSON array containing the extracted subject entities. The output should follow this structure:\n["entity_1", "entity_2", "entity_3", ...]''',
        "fact_list_template": '''You are a professional relationship extraction expert. Given a list of relations: "{relation_list}" and a document: "{sentences}", along with a head entity "{head}", extract all triplet facts from the document where the head entity is the subject. For each triplet, identify the relation and the corresponding object. Output the result in JSON format as follows:\n{{rel1:[object1,object2,...],rel2:[object1,object2,...],...}}'''
    },
        "summery": {
            "summery_template": '''you are a document theme summarization expert. Please summarize a theme for this document based on the one I provide.'''
    },

}
