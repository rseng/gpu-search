# https://github.com/RWTH-EBC/FiLiP

```console
examples/ngsi_v2/e11_ngsi_v2_semantics/ontology_files/sargon.ttl:                                owl:allValuesFrom :OpenCloseState
examples/ngsi_v2/e11_ngsi_v2_semantics/ontology_files/sargon.ttl:            rdfs:subClassOf :OpenCloseState ;
examples/ngsi_v2/e11_ngsi_v2_semantics/ontology_files/sargon.ttl:                              owl:someValuesFrom :OpenCloseFunction
examples/ngsi_v2/e11_ngsi_v2_semantics/ontology_files/sargon.ttl:                              owl:someValuesFrom :OpenCloseState
examples/ngsi_v2/e11_ngsi_v2_semantics/ontology_files/sargon.ttl:            rdfs:comment "A device of category saref:Actuator that consists of a switch, accomplishes the task saref:Safety, performs the saref:OpenCloseFunction, is used for controlling a door, and can be found in the state saref:OpenCloseState."^^xsd:string ;
examples/ngsi_v2/e11_ngsi_v2_semantics/ontology_files/sargon.ttl:###  https://w3id.org/saref#OpenCloseFunction
examples/ngsi_v2/e11_ngsi_v2_semantics/ontology_files/sargon.ttl::OpenCloseFunction rdf:type owl:Class ;
examples/ngsi_v2/e11_ngsi_v2_semantics/ontology_files/sargon.ttl:###  https://w3id.org/saref#OpenCloseState
examples/ngsi_v2/e11_ngsi_v2_semantics/ontology_files/sargon.ttl::OpenCloseState rdf:type owl:Class ;
examples/ngsi_v2/e11_ngsi_v2_semantics/ontology_files/sargon.ttl:                               owl:allValuesFrom :OpenCloseState
examples/ngsi_v2/e11_ngsi_v2_semantics/ontology_files/sargon.ttl:           rdfs:subClassOf :OpenCloseState ;
examples/ngsi_v2/e11_ngsi_v2_semantics/ontology_files/sargon.ttl:        rdfs:comment "A device of category saref:Actuator that performs an actuating function of type saref:OnOffFunction or saref:OpenCloseFunction"^^xsd:string ;

```