"""Riemann-type approximation with an explicitly attributed MVDC evaluator."""
import argparse
from mvdc_prime_count import (DEFAULT_ORDER, DEFAULT_PREFIX_CAP, MAX_REFERENCE_X,
                              approximate_pi, approximate_pi_auto, format_result)


def create_app():
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext
    root = tk.Tk()
    root.title('Prime-count approximation: Riemann model + MVDC evaluation')
    frame = ttk.Frame(root, padding=12)
    frame.grid(sticky='nsew')
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)
    fields = {}
    specs = (('x', 'x', '1000000'), ('alpha', 'Requested prefix exponent alpha', '.9'),
             ('max_cutoff', 'Maximum exact prefix (work budget)', str(DEFAULT_PREFIX_CAP)),
             ('order', 'Even MVDC numerical order', str(DEFAULT_ORDER)),
             ('K_factor', 'Classical root-model K factor', '2'), ('cutoff', 'Optional explicit prefix M', ''))
    for row, (key, label, initial) in enumerate(specs):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky='w', pady=3)
        entry = ttk.Entry(frame)
        entry.insert(0, initial)
        entry.grid(row=row, column=1, sticky='ew')
        fields[key] = entry
    verify = tk.BooleanVar(value=True)
    ttk.Checkbutton(frame, text='Compute exact pi(x) separately for comparison when x <= 10^7', variable=verify).grid(row=6, columnspan=2, sticky='w')
    output = scrolledtext.ScrolledText(frame, state='disabled', width=110, height=20)
    output.grid(row=8, columnspan=2, sticky='nsew', pady=8)
    frame.rowconfigure(8, weight=1)
    def compute():
        try:
            x = int(fields['x'].get())
            cutoff = fields['cutoff'].get().strip()
            result = approximate_pi(x, alpha=float(fields['alpha'].get()),
                max_cutoff=int(fields['max_cutoff'].get()), order=int(fields['order'].get()),
                K_factor=float(fields['K_factor'].get()), cutoff=int(cutoff) if cutoff else None,
                return_details=True)
            message = format_result(result, verify.get() and x <= MAX_REFERENCE_X)
            if verify.get() and x > MAX_REFERENCE_X:
                message += '\nLocal exact comparison skipped; published large-x references are in the report.'
        except (ValueError, OverflowError) as error:
            messagebox.showerror('Invalid approximation input', str(error), parent=root)
            return
        output.configure(state='normal')
        output.delete('1.0', tk.END)
        output.insert(tk.END, message)
        output.configure(state='disabled')
    ttk.Button(frame, text='Compute approximation', command=compute).grid(row=7, columnspan=2, pady=6)
    root.geometry('1050x720')
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--x', type=int, help='Run without a GUI')
    parser.add_argument('--order', type=int, default=DEFAULT_ORDER)
    parser.add_argument('--alpha', type=float, default=.9)
    parser.add_argument('--k-factor', type=float, default=2)
    parser.add_argument('--cutoff', type=int)
    parser.add_argument('--max-cutoff', type=int, default=DEFAULT_PREFIX_CAP)
    parser.add_argument('--numerical-tolerance', type=float, help='Tail evaluation only, not total count error')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.x is None:
        create_app().mainloop()
        return
    kwargs = dict(alpha=args.alpha, K_factor=args.k_factor, cutoff=args.cutoff,
                  max_cutoff=args.max_cutoff, return_details=True)
    try:
        if args.numerical_tolerance is None:
            result = approximate_pi(args.x, order=args.order, **kwargs)
        else:
            result = approximate_pi_auto(args.x, args.numerical_tolerance, **kwargs)
        print(format_result(result, args.verify))
    except ValueError as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
