if [ $# -lt 1 ]; then
  echo "Usage $0 <H5 path>"
fi

nice -n 19 matlab -nodisplay -r "cd ../utils/; convert_mpii_result_for_eval('$1'); exit;"
