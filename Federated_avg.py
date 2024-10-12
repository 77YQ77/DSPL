
def federated_average(model_list, coefficient_matrix, batchnorm_mmd=True):
    """
    :param model_list: a list of all models needed in federated average. [0]: model for target domain,
    [1:-1] model for source domains
    :param coefficient_matrix: the coefficient for each model in federate average, list or 1-d np.array
    :param batchnorm_mmd: bool, if true, we use the batchnorm mmd
    :return model list after federated average
    """
    if batchnorm_mmd:
        dict_list = [it.state_dict() for it in model_list]
        dict_item_list = [dic.items() for dic in dict_list]
        for key_data_pair_list in zip(*dict_item_list):
            source_data_list = [pair[1] * coefficient_matrix[idx] for idx, pair in
                                enumerate(key_data_pair_list)]
            dict_list[0][key_data_pair_list[0][0]] = sum(source_data_list)
        for model in model_list:
            model.load_state_dict(dict_list[0])
    else:
        named_parameter_list = [model.named_parameters() for model in model_list]
        for parameter_list in zip(*named_parameter_list):
            source_parameters = [parameter[1].data.clone() * coefficient_matrix[idx] for idx, parameter in
                                 enumerate(parameter_list)]
            parameter_list[0][1].data = sum(source_parameters)
            for parameter in parameter_list[1:]:
                parameter[1].data = parameter_list[0][1].data.clone()