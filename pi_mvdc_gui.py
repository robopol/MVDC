"""Prime-count research tool: exact finite front and explicit tail models.

Importing this module never creates a window. Use --x for a headless CLI run.
The former unsupported MVDC-to-prime-count conversion is no longer used.
"""
import argparse
from mvdc_prime_count import (MAX_EXACT_CUTOFF, approximate_pi, approximate_pi_auto,
                              format_result, mvdc_block, primes_up_to, select_cutoff)


def create_app():
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext

    root = tk.Tk()
    root.title("Prime counting: exact front and modelled tail")
    frame = ttk.Frame(root, padding=12)
    frame.grid(sticky="nsew")
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(6, weight=1)
    fields = {}
    for row, (key, label, value) in enumerate((
        ("x", "x (2 to 10^18)", "1000000"),
        ("target", "Effort target (not an error guarantee)", "1e-4"),
        ("cap", f"Maximum exact cutoff (up to {MAX_EXACT_CUTOFF})", "1000000"),
    )):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=4)
        entry = ttk.Entry(frame)
        entry.insert(0, value)
        entry.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        fields[key] = entry
    ttk.Label(frame, text="Tail model").grid(row=3, column=0, sticky="w")
    model = ttk.Combobox(frame, values=("riemannr", "li"), state="readonly")
    model.set("riemannr")
    model.grid(row=3, column=1, sticky="ew")
    verify = tk.BooleanVar(value=False)
    ttk.Checkbutton(frame, text=f"Compare with exact count when x <= {MAX_EXACT_CUTOFF}",
                    variable=verify).grid(row=4, column=0, columnspan=2, sticky="w")
    output = scrolledtext.ScrolledText(frame, state="disabled", width=85, height=15)
    output.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=8)

    def compute():
        try:
            result = approximate_pi_auto(int(fields["x"].get()),
                                         float(fields["target"].get().replace(",", ".")),
                                         max_cutoff=int(fields["cap"].get()),
                                         model=model.get(), return_details=True)
            text = format_result(result, verify.get())
        except (ValueError, OverflowError) as exc:
            messagebox.showerror("Invalid input", str(exc), parent=root)
            return
        output.configure(state="normal")
        output.delete("1.0", tk.END)
        output.insert(tk.END, text)
        output.configure(state="disabled")

    ttk.Button(frame, text="Compute", command=compute).grid(row=5, column=0, columnspan=2, pady=8)
    root.geometry("900x520")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=int, help="Run the numerical backend without opening a GUI")
    parser.add_argument("--cutoff", type=int)
    parser.add_argument("--target", type=float, default=1e-4)
    parser.add_argument("--model", choices=("riemannr", "li"), default="riemannr")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.x is None:
        create_app().mainloop()
        return
    result = (approximate_pi(args.x, cutoff=args.cutoff, model=args.model, return_details=True)
              if args.cutoff is not None else
              approximate_pi_auto(args.x, args.target, model=args.model, return_details=True))
    print(format_result(result, args.verify))


if __name__ == "__main__":
    main()
