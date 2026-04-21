"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Loader2, Trash2, Upload, FileText } from "lucide-react"
import {
  deleteKnowledgeDocument,
  getKnowledgeDocuments,
  getOrgId,
  uploadKnowledgePdf,
  type KnowledgeDocument,
} from "@/lib/api"
import { cn } from "@/lib/utils"

function statusBadgeClass(status: KnowledgeDocument["status"]) {
  switch (status) {
    case "ready":
      return "bg-emerald-100 text-emerald-800 border-emerald-200 dark:bg-emerald-950 dark:text-emerald-200 dark:border-emerald-800"
    case "processing":
      return "bg-amber-100 text-amber-900 border-amber-200 dark:bg-amber-950 dark:text-amber-100 dark:border-amber-800"
    case "failed":
      return "bg-red-100 text-red-800 border-red-200 dark:bg-red-950 dark:text-red-200 dark:border-red-800"
    default:
      return ""
  }
}

function formatStatusLabel(status: KnowledgeDocument["status"]) {
  switch (status) {
    case "ready":
      return "Processed"
    case "processing":
      return "Processing"
    case "failed":
      return "Failed"
    default:
      return status
  }
}

export default function KnowledgeBasePage() {
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([])
  const [isLoadingList, setIsLoadingList] = useState(true)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [pendingDelete, setPendingDelete] = useState<KnowledgeDocument | null>(
    null
  )
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const loadDocuments = useCallback(async () => {
    try {
      const list = await getKnowledgeDocuments()
      setDocuments(list)
    } catch (e) {
      console.error(e)
    } finally {
      setIsLoadingList(false)
    }
  }, [])

  useEffect(() => {
    loadDocuments()
  }, [loadDocuments])

  const hasProcessing = documents.some((d) => d.status === "processing")

  useEffect(() => {
    if (!hasProcessing) return
    const id = window.setInterval(() => {
      loadDocuments()
    }, 3000)
    return () => window.clearInterval(id)
  }, [hasProcessing, loadDocuments])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    setUploadError(null)
    if (!file) {
      setSelectedFile(null)
      return
    }
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setSelectedFile(null)
      setUploadError("Please choose a PDF file.")
      return
    }
    setSelectedFile(file)
  }

  const handleProceed = async () => {
    const orgId = getOrgId()
    if (!orgId || !selectedFile) return
    setIsUploading(true)
    setUploadError(null)
    try {
      await uploadKnowledgePdf(selectedFile, orgId)
      setSelectedFile(null)
      if (fileInputRef.current) fileInputRef.current.value = ""
      await loadDocuments()
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed")
    } finally {
      setIsUploading(false)
    }
  }

  const handleConfirmDelete = async () => {
    if (!pendingDelete) return
    setDeleteError(null)
    setDeletingId(pendingDelete.document_id)
    try {
      await deleteKnowledgeDocument(pendingDelete.document_id)
      setPendingDelete(null)
      await loadDocuments()
    } catch (err) {
      setDeleteError(
        err instanceof Error ? err.message : "Could not delete document"
      )
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="border-b border-slate-200 bg-white px-6 py-5 shrink-0">
        <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">
          Knowledge Base
        </h1>
      </div>

      <div className="flex-1 overflow-auto px-6 py-6 space-y-6">
        <Card className="max-w-2xl border-slate-200">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Upload className="h-5 w-5 text-slate-600" />
              Upload document
            </CardTitle>
            <CardDescription>
              Select a text-based PDF, then click Proceed to start processing.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center gap-3">
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,application/pdf"
                className="hidden"
                id="kb-pdf-input"
                onChange={handleFileChange}
              />
              <label htmlFor="kb-pdf-input">
                <Button type="button" variant="outline" asChild className="cursor-pointer">
                  <span>Choose PDF</span>
                </Button>
              </label>
              <span className="text-sm text-muted-foreground truncate flex items-center gap-2 min-w-0">
                {selectedFile ? (
                  <>
                    <FileText className="h-4 w-4 shrink-0" />
                    <span className="truncate">{selectedFile.name}</span>
                  </>
                ) : (
                  "No file selected"
                )}
              </span>
            </div>
            {uploadError && (
              <p className="text-sm text-red-600" role="alert">
                {uploadError}
              </p>
            )}
            <Button
              type="button"
              disabled={!selectedFile || isUploading}
              onClick={handleProceed}
              className="min-w-[120px]"
            >
              {isUploading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Uploading…
                </>
              ) : (
                "Proceed"
              )}
            </Button>
          </CardContent>
        </Card>

        <Card className="border-slate-200">
          <CardHeader>
            <CardTitle className="text-lg">Your knowledge files</CardTitle>
            <CardDescription>
              Status updates automatically while files are processing.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoadingList ? (
              <div className="flex items-center gap-2 text-muted-foreground py-8 justify-center">
                <Loader2 className="h-5 w-5 animate-spin" />
                Loading…
              </div>
            ) : documents.length === 0 ? (
              <p className="text-sm text-muted-foreground py-8 text-center">
                No documents yet. Upload a PDF above.
              </p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>File</TableHead>
                    <TableHead className="w-[140px]">Status</TableHead>
                    <TableHead className="min-w-[160px]">Updated</TableHead>
                    <TableHead className="w-[100px] text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {documents.map((doc) => (
                    <TableRow key={doc.document_id}>
                      <TableCell className="font-medium max-w-[280px] truncate">
                        {doc.original_filename}
                        {doc.status === "failed" && doc.error_message && (
                          <span
                            className="block text-xs text-red-600 font-normal mt-1 truncate"
                            title={doc.error_message}
                          >
                            {doc.error_message}
                          </span>
                        )}
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant="outline"
                          className={cn("font-normal", statusBadgeClass(doc.status))}
                        >
                          {doc.status === "processing" && (
                            <Loader2 className="h-3 w-3 animate-spin mr-1 inline" />
                          )}
                          {formatStatusLabel(doc.status)}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-muted-foreground text-sm whitespace-nowrap">
                        {doc.updated_at
                          ? new Date(doc.updated_at).toLocaleString()
                          : "—"}
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          className="text-red-600 hover:text-red-700 hover:bg-red-50"
                          aria-label={`Delete ${doc.original_filename}`}
                          onClick={() => {
                            setDeleteError(null)
                            setPendingDelete(doc)
                          }}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </div>

      <AlertDialog
        open={pendingDelete != null}
        onOpenChange={(open) => {
          if (!open && !deletingId) {
            setPendingDelete(null)
            setDeleteError(null)
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete document?</AlertDialogTitle>
            <AlertDialogDescription>
              This removes{" "}
              <span className="font-medium text-foreground">
                {pendingDelete?.original_filename}
              </span>{" "}
              and its indexed content. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          {deleteError && (
            <p className="text-sm text-red-600" role="alert">
              {deleteError}
            </p>
          )}
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deletingId != null}>
              Cancel
            </AlertDialogCancel>
            <Button
              type="button"
              variant="destructive"
              disabled={deletingId != null}
              onClick={handleConfirmDelete}
            >
              {deletingId != null ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Deleting…
                </>
              ) : (
                "Delete"
              )}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
