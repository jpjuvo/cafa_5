
# This class is copied from https://www.kaggle.com/code/mtinti/merge-datasets
# mtinti

# Define a class to manage predictions for proteins.
# The class keeps track of the highest score for each GO (Gene Ontology) term prediction.
# Note: This assumes scores are comparable, which might not be the case.
# A ranking-based selection could be more suitable.
# Each branch outputs a maximum of 35 predictions for each protein after sorting predictions from highest to lowest.
# There is an option to add a bonus to the score if the term is predicted by multiple methods.

class ProteinPredictions:
    # Initialize an empty dictionary to store the predictions
    def __init__(self):
        self.predictions = {}

    # Add a prediction to the storage, with optional bonus
    # Arguments:
    #   - protein: Identifier for the protein
    #   - go_term: GO term that is being predicted
    #   - score: Confidence score of the prediction
    #   - branch: Branch of the Gene Ontology (e.g., 'CCO', 'MFO', 'BPO')
    #   - bonus: Optional bonus to be added to the score
    def add_prediction(self, protein, go_term, score, branch, bonus=0, hist=0, inc=1):
        # If the protein is not already in the storage, initialize its structure
        if protein not in self.predictions:
            self.predictions[protein] = {'CCO': {}, 'MFO': {}, 'BPO': {}}
        
        # Convert the score to a float for comparison and calculation
        score = float(score)

        # If this GO term has already been predicted for this protein and branch,
        # add the bonus to the score. Keep the highest score.
        if go_term in self.predictions[protein][branch]:
            if self.predictions[protein][branch][go_term] < score:
                self.predictions[protein][branch][go_term] = (score*inc+self.predictions[protein][branch][go_term]*hist)/(hist+inc)    + bonus
            else:
                self.predictions[protein][branch][go_term]  = (score*inc+self.predictions[protein][branch][go_term]*hist)/(hist+inc) + bonus
        # If this GO term has not been predicted yet, store it with the score
        else:
            self.predictions[protein][branch][go_term] = score

        # Ensure that the score does not exceed 1
        if self.predictions[protein][branch][go_term] > 1:
            self.predictions[protein][branch][go_term] = 1

    # Export the stored predictions to a file
    # Arguments:
    #   - output_file: File name for the exported predictions
    #   - top: Number of top predictions to export for each protein and branch
    def get_predictions(self, output_file='submission.tsv', top=42):
        # Open the output file
        with open(output_file, 'w') as f:
            # Iterate through each protein and its branches
            for protein, branches in self.predictions.items():
                # For each branch, sort the GO terms by score in descending order and select the top ones
                for branch, go_terms in branches.items():
                    # Sort go_terms by score in descending order and take the top ones
                    top_go_terms = sorted(go_terms.items(), key=lambda x: x[1], reverse=True)[:top]
                    # Write each of the top predictions to the file
                    for go_term, score in top_go_terms:
                        f.write(f"{protein}\t{go_term}\t{score:.3f}\n")