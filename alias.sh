alias av="python -m avcv.run"

alias copy-unzip="av copy_unzip -a"
dpython(){
    ori_cmd="python "$@""
    echo $ori_cmd
    av parse_debug_command -a $ori_cmd
}

vs_run(){
    ori_cmd="$@"
    av change_vscode_run_cmd -a $ori_cmd
}

alias v2i="av video_to_images  -a"
alias i2v="av images_to_video  -a"
