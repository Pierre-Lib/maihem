#Remove all folders with previous results to avoid confusion and buildup
rm -rf predict* test_model*

#Ask user for input to select the example version
echo "Select which example to run (1-4):"
echo "1) No input, only checking if the binaries work"
echo "2) Training only, input: example_2_input.json"
echo "3) Usage only, input: example_3_input.json"
echo "4) Training and usage; complete run. Input: example_4_input.json"

#Read user input
read -p "Enter a number (1-4): " choice

#Assign input file depending on input
case $choice in
  1) input_file="" ;;
  2) input_file="example_2_input.json" ;;
  3) input_file="example_3_input.json" ;;
  4) input_file="example_4_input.json" ;;
  *) echo "Please enter a number between 1 and 4; exiting" && exit 1 ;;
esac

#Run the package according to the input
if [[ -z "$input_file" ]]; then
  maihem
else
  maihem -i "$input_file"
fi

echo "Run completed. Thank you for trying maihem!"



