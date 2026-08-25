export default function AuthLoading() {
  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-6 p-8">
      <div className="route-skeleton h-10 w-32 rounded-lg" />
      <div className="w-full max-w-md space-y-4">
        <div className="route-skeleton h-12 w-full rounded-lg" />
        <div className="route-skeleton h-12 w-full rounded-lg" />
        <div className="route-skeleton h-11 w-full rounded-lg" />
      </div>
    </div>
  )
}
