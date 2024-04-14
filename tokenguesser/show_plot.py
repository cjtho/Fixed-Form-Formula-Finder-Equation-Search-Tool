import json

from matplotlib import pyplot as plt

with open("training_data.json", "r") as f:
    data = json.load(f)


def plot_data(recorded_data) -> None:
    loss_points = []
    curriculum_points = []
    all_iterations = []

    for curriculum, records in recorded_data.items():
        for iteration, data in records:
            min_loss, mean_loss, max_loss = data["loss"]
            loss_points.append((iteration, mean_loss, min_loss, max_loss))
            all_iterations.append(iteration)

        if records:
            curriculum_points.append((records[0][0], float(curriculum)))

    iterations, mean_losses, min_losses, max_losses = zip(*loss_points)

    plt.figure(figsize=(10, 6))

    loss_error_bars = [[mean_loss - min_loss for mean_loss, min_loss in zip(mean_losses, min_losses)],
                       [max_loss - mean_loss for max_loss, mean_loss in zip(max_losses, mean_losses)]]

    plt.errorbar(iterations, mean_losses, yerr=loss_error_bars, fmt="-o", label="Loss", capsize=5)

    placed = False
    for time_step, curriculum in curriculum_points:
        if not placed:
            plt.axvline(x=time_step, color="red", linestyle="-", linewidth=1.5, ymin=0, ymax=999, label="Curriculum")
            placed = True
        else:
            plt.axvline(x=time_step, color="red", linestyle="-", linewidth=1.5, ymin=0, ymax=999)
        plt.text(time_step, plt.ylim()[1] * 0.95, f"{float(curriculum):.3f}",
                 rotation=0,
                 verticalalignment="top",
                 color="red")

    plt.title("Training Loss and Metric Over Iterations with Curriculums")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()


plot_data(data)
