import { Dashboard } from "@/components/Dashboard";

export default function Home() {
  return (
    <>
      <div className="sticky top-0 z-10 bg-amber-100 px-4 py-2 text-center text-sm text-amber-900">
        Synthetic data. Every number here is simulation output from a synthetic payment ecosystem,
        not real payment performance, and not Razorpay&apos;s.
      </div>
      <Dashboard />
    </>
  );
}
