#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""desc"""


from __future__ import unicode_literals

import os
import json
import argparse
import glob
import yaml
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import bs4
pd.set_option('display.max_colwidth', -1)
from nct.nlp import tokenizer

from itertools import repeat

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.inputters.text_dataset import ContinuousField
from torchtext.data import Field


def str2bool(v: str) -> bool:
    """
    str to bool
    """
    return v.lower() in ("yes", "true", "t", "1")


def config_demo_parser(parser):
    """
    config server parser
    """
    group = parser.add_argument_group('Demo')
    group.add_argument("-p", "--port", help="port", default=5120)
    supported_tokenizer_classes = str([k for (k, v) in tokenizer.__dict__.items() if str(v).startswith('<class') and k.endswith('Tokenizer') and not k.startswith('Tokenizer')])
    group.add_argument("--tokenizer_class", "-tc",  help="tokenizer class named used in preprocessing, supported classes are {}".format(supported_tokenizer_classes), default="SplitTokenizer", type=str)
    # group.add_argument("-csmf", "--control-signal-mapping-file", help="path of csv file, which has control_signal_name, key, value columns, key -mapped->value in frontend", default=None)
    group.add_argument("--control_signal_name_mapping", "-csnm", help="path of csv file, which has control_signal_name, key, value columns, key -mapped->value in frontend", default=None, type=str)


def train_opts(parser):
    """ Training and saving options """

    group = parser.add_argument_group('General')


def get_parser():
    """
    get the parser for findlr/train/translate
    """
    main_arg_parser = argparse.ArgumentParser("onmt", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = main_arg_parser.add_subparsers(description="valid subcommand", dest="subcommand")

    # translate-demo
    demo_arg_parser = subparsers.add_parser("demo", help="start a translate http server for demo", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config_demo_parser(demo_arg_parser)

    return main_arg_parser


def demo(args):
    """
    start a translate server for demo

    todo:
    - 支持控制信号映射: 保证可读性 (done)
      训练模型时, 部分控制信号用的是str(数字)，比如类目ID(没有采用类目名称，保证类目的唯一性)，但是在html中，展示类目ID不直观

    """

    # control_signal_mapping = {}  # control_signal_name -> {key:value}
    # if args.control_signal_mapping_file:
    #     df_cs_mapping = pd.read_csv(args.control_signal_mapping_file, sep="\001")
    #     for control_signal_name in set(df_cs_mapping["control_signal_name"].values):
    #         tmp_df = df_cs_mapping.loc[df_cs_mapping["control_signal_name"] == control_signal_name]
    #         control_signal_mapping[control_signal_name] = {str(row["key"]): str(row["value"]) for idx, row in tmp_df.iterrows()}

    def update_input_html(html, selected_info):
        """
        update html using selected info without control_signal part
        - update <div name="input_text">
        - update <div name="target_text">
        - update <div name="predict_parameter">
        """
        if selected_info:
            soup = bs4.BeautifulSoup(html, "lxml")

            # update <div> of input_text
            if "src_sentence" in selected_info:
                input_textarea = soup.find("div", {"name": "input_text"}).find("textarea", {"name": "src_sentence"})
                input_textarea.insert(0, selected_info["src_sentence"])

            # update <div> of target_text
            if "tgt_sentence" in selected_info:
                input_textarea = soup.find("div", {"name": "target_text"}).find("textarea", {"name": "tgt_sentence"})
                input_textarea.insert(0, selected_info["tgt_sentence"])

            # update <div> of predict_parameter
            div_predict_parameter = soup.find("div", {"name": "predict_parameter"})
            input_tags = div_predict_parameter.find_all("input")
            for t in input_tags:
                name = t.attrs["name"]
                if name in selected_info:
                    if t.attrs["type"] != "radio":
                        t.attrs["value"] = selected_info[name]
                    else:
                        t.attrs = {k: v for (k, v) in t.attrs.items() if k != "checked"}
                        if selected_info[name] == t.attrs["value"]:
                            t.attrs["checked"] = ""

            result = soup.prettify()
            return result

        return html

    def update_control_signal_html(html, selected_info):
        """
        update html using the seleted_info:
        - update <select>: selected
        - update <input>: value
        """
        if selected_info:
            soup = bs4.BeautifulSoup(html, "lxml")

            # update <select> tags
            select_tags = soup.find_all("select")
            for t in select_tags:
                name = t.attrs["name"]
                if name in selected_info:
                    options = t.find_all("option")

                    for option in options:
                        if option.attrs["value"] == selected_info[name]:
                            option.attrs.update({"selected": ""})
                            break

            # update <input> tags
            input_tags = soup.find_all("input")
            for t in input_tags:
                name = t.attrs["name"]
                if name in selected_info:
                    t.attrs["value"] = selected_info[name]

            result = soup.prettify()
            return result

        return html

    logger = init_logger(opt.log_file)

    translator = build_translator(opt, report_score=True)

    def init_control_signal_html(fields, control_signal_name_mapping=None):
        if control_signal_name_mapping is None:
            control_signal_name_mapping = {}

        lis = ""
        control_signal_info = []    # name -> values
        for name, field in fields['tgt'].fields:
            if name.startswith("tgt_ctrl_signal"):
                display_name = control_signal_name_mapping[name] if name in control_signal_name_mapping else name
                if isinstance(field, ContinuousField):
                    lis += """<li>{name} <input type="number" name="{name}" value=""></input></li>\n""".format(name=display_name)
                    control_signal_info.append((name, display_name, "continuous"))
                else:
                    option_values = "\n".join(["""<option value="{name}"> {display_value} </option>""".format(name=e, display_value=e) for e in field.vocab.itos if not e.startswith("<")])
                    lis += """<li>{name} <select name={name} class="selectpicker show-tick form-control" data-live-search="true">\n {option_values} \n</select></li>\n""".format(name=display_name, option_values=option_values)
                    control_signal_info.append((name, display_name, "categorical"))

        html = """<div name="control_signal"><p>控制信号<br></p><ul>\n{}\n</ul></div>""".format(lis)
        return html, control_signal_info

    default_control_signal_html, control_signal_info = init_control_signal_html(translator.fields, eval(args.control_signal_name_mapping))
    the_tokenizer = getattr(tokenizer, args.tokenizer_class)()

    app = Flask(__name__)

    @app.route("/", methods=["POST", "GET"])
    def nmt():
        """
        todoc
        """
        if request.method == "GET":
            return render_template("input.html", control_signal_html=default_control_signal_html)

        if request.method == "POST":
            raw_src_sentence = request.form["src_sentence"]
            raw_tgt_sentence = request.form["tgt_sentence"]
            src_sentence = " ".join(the_tokenizer.tokenize(raw_src_sentence))
            tgt_sentence = " ".join(the_tokenizer.tokenize(raw_tgt_sentence)) if raw_tgt_sentence else None

            n_best = int(request.form["n_best"])
            beam_size = int(request.form["beam_size"])
            block_n_gram = int(request.form["block_n_gram"])
            replace_unk = str2bool(request.form["replace_unk"])
            translator.n_best = n_best
            translator.beam_size = beam_size
            translator.block_ngram_repeat = block_n_gram
            translator.replace_unk = replace_unk

            # using external variable: config
            candidate_control_signal_names = [ee for e in control_signal_info for ee in (e[0], e[1])]  # 顺序很重要
            ctrl_signal_str = ""
            ctrl_signal_dict = {}
            for (k, v) in request.form.items():
                if k in candidate_control_signal_names:
                    for n1, n2, _type in control_signal_info:
                        if k == n1 or k == n2:
                            if _type == "categorical":
                                ctrl_signal_str += "￨" + str(v)
                            else:
                                ctrl_signal_str += "￫" + str(v)

                            ctrl_signal_dict[n1] = v

            src_path = "/tmp/onmt_demo_src.txt"
            tgt_path = "/tmp/onmt_demo_tgt.txt"
            if raw_tgt_sentence:
                tgt_row = " ".join([e + ctrl_signal_str for e in the_tokenizer.tokenize(raw_tgt_sentence)])
                with open(tgt_path, "w") as f:
                    f.write(tgt_row + "\n")
            else:
                tgt_path = None
                tgt_row = None

            src_row = " ".join([e for e in the_tokenizer.tokenize(raw_src_sentence)])
            with open(src_path, "w") as f:
                f.write(src_row + "\n")

            predict_params = {
                "src": src_row,
                "tgt": tgt_row,
                "beam_size": beam_size,
                "n_best": n_best,
                "block_n_gram": block_n_gram,
                "replace_unk": replace_unk,
                "control_signal_dict": ctrl_signal_dict,
            }
            # predict_params_html = pd.DataFrame({k: str(v) if isinstance(v, dict) else v for (k, v) in predict_params.items()}, index=["输入参数值"]).T.to_html()
            predict_params_str = "<br>".join(["{}:\t{}".format(e[0], e[1]) for e in predict_params.items()])

            src_shards = split_corpus(src_path, shard_size=-1)
            tgt_shards = split_corpus(tgt_path, shard_size=-1) if tgt_path is not None else repeat(None)
            shard_pairs = zip(src_shards, tgt_shards)

            translator.n_best = n_best
            translator.beam_size = beam_size
            translator.block_n_gram = block_n_gram
            translator.replace_unk = replace_unk
            for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
                logger.info("Translating shard %d." % i)
                all_scores, all_predictions = translator.translate(
                    src=src_shard,
                    tgt=tgt_shard,
                    src_dir=opt.src_dir,
                    batch_size=opt.batch_size,
                    attn_debug=opt.attn_debug,
                    control_signal_dict=ctrl_signal_dict
                )

            def remove_bad_ending(prediction):
                n_ending = 0
                for word in prediction.split()[::-1]:
                    if word in "。，！的":
                        n_ending += 1
                    else:
                        break
                if n_ending >= 2:
                    result = " ".join(prediction.split()[:-(n_ending - 1)])
                    print("{} -> \n{}".format(prediction, result))
                else:
                    result = prediction

                return result
                # return prediction

            all_scores = [e.item() for e in all_scores[0]]
            all_predictions = [remove_bad_ending(e) for e in all_predictions[0]]

            df_result = pd.DataFrame({"score": all_scores, "prediction": all_predictions})
            df_result_str = df_result.to_html(index=False)

            return update_input_html(render_template("output.html",
                                                     predict_result=df_result_str,
                                                     predict_params=predict_params_str,
                                                     control_signal_html=update_control_signal_html(default_control_signal_html, request.form)), request.form)

        return "only GET/POST method suportted"

    app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
    app.run(host='0.0.0.0', port=args.port, debug=False)


def _get_parser():
    parser = ArgumentParser(description='demo')
    opts.config_opts(parser)
    opts.translate_opts(parser)
    config_demo_parser(parser)

    return parser


if __name__ == '__main__':

    parser = _get_parser()
    opt = parser.parse_args()
    demo(opt)
